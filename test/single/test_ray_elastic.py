"""Ray-Horovod Elastic training unit tests.

This is currently not run on the Ray CI.
"""
from contextlib import contextmanager
import psutil
import os
import socket
import time
import mock
import pytest
import ray
from horovod.common.util import gloo_built
from horovod.runner.elastic.discovery import HostDiscovery
from horovod.ray.elastic import ElasticRayExecutor, RayHostDiscovery


@pytest.fixture
def ray_shutdown():
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_8_cpus():
    ray.init(num_cpus=8, resources={
        f"node:host-{i}": 1 for i in range(10)})
    yield
    # The code after the yield will run as teardown code.
    ray.shutdown()


@pytest.fixture
def ray_8_cpus_gpus():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if len(set(os.environ["CUDA_VISIBLE_DEVICES"].split(","))) < 8:
            pytest.skip("Avoiding mismatched GPU machine.")
    ray.init(num_cpus=8, num_gpus=8, resources={
        f"node:host-{i}": 1 for i in range(10)})
    try:
        yield
    finally:
        # The code after the yield will run as teardown code.
        ray.shutdown()


class TestRayDiscoverySuite:
    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_cpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [4]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_gpu_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=1)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [1]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_gpu_slot_discovery(self, ray_shutdown):
        ray.init(num_cpus=4, num_gpus=4)
        discovery = RayHostDiscovery(
            use_gpu=True, cpus_per_slot=1, gpus_per_slot=2)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 1
        assert list(mapping.values()) == [2]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_multinode(self, monkeypatch):
        def create_multi_node_mock():
            host_names = ["host-1", "host-2", "host-3"]
            resources = {"GPU": 2, "CPU": 8}

            def create_node_entry(hostname):
                return {
                    "NodeManagerAddress": hostname,
                    "Resources": resources.copy(),
                    "alive": True
                }

            return map(create_node_entry, host_names)

        monkeypatch.setattr(ray, "nodes", create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 3
        assert list(mapping.values()) == [2, 2, 2]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_multinode_gpus_per_slot(self, monkeypatch):
        def create_multi_node_mock():
            host_names = ["host-1", "host-2", "host-3"]
            resources = {"GPU": 2, "CPU": 8}

            def create_node_entry(hostname):
                return {
                    "NodeManagerAddress": hostname,
                    "Resources": resources.copy(),
                    "alive": True
                }

            return map(create_node_entry, host_names)

        monkeypatch.setattr(ray, "nodes", create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, gpus_per_slot=2)
        mapping = discovery.find_available_hosts_and_slots()
        assert len(mapping) == 3
        assert list(mapping.values()) == [1, 1, 1]

    @pytest.mark.skipif(
        not gloo_built(), reason='Gloo is required for Ray integration')
    def test_multinode_mismatch(self, monkeypatch):
        def create_multi_node_mock():
            host_names = ["host-1", "host-2", "host-3"]
            resources = {"CPU": 8}

            def create_node_entry(hostname):
                return {
                    "NodeManagerAddress": hostname,
                    "Resources": resources.copy(),
                    "alive": True
                }

            return map(create_node_entry, host_names)

        monkeypatch.setattr(ray, "nodes", create_multi_node_mock)
        discovery = RayHostDiscovery(use_gpu=True, cpus_per_slot=1)
        mapping = discovery.find_available_hosts_and_slots()
        assert sum(mapping.values()) == 0


class SimpleTestDiscovery(HostDiscovery):
    def __init__(self, schedule, wait_for_previous_set=True):
        self._schedule = schedule
        self._generator = self.host_generator()
        self.executor = None
        # The previous set of hosts
        # We need a reference to them as iterators only can provide the next set
        self.prevlist = None
        self.wait_for_previous_set = wait_for_previous_set

    def host_generator(self):
        for iters, hosts in self._schedule:
            iters = iters or 500  # max
            for i in range(iters):
                yield hosts

    def find_available_hosts_and_slots(self):
        hostlist = next(self._generator)
        # Ensure discovery waits for the previous set to register
        self._wait_for_previous_set_registration(hostlist)

        hosts = {}
        for item in hostlist:
            host, slots = item.split(":")
            slots = int(slots)
            hosts[host] = slots

        return hosts

    def _wait_for_previous_set_registration(self, hostlist):
        """
        Ensure that at least one host from the previous set of hosts have
        been registered.
        Without this, the discovery script will "discover" the new
        set of hosts before the current set can register.
        This would result in a race condition.
        Consider a discovery schedule:
        ```
        discovery_schedule = [
            (10, ['host-1:2']),
            (30, ['host-1:2', 'host-2:1', 'host-3:1']),
            (None, ['host-2:1']),
        ]
        ```
        The initial set is: ['host-1:2']. Before this is registered in the driver, the discovery script
        discovers the set: ['host-1:2', 'host-2:1', 'host-3:1'], and adds ['host-2:1', 'host-3:1'].
        However, since ['host-1:2'] has not registered, there is no coordinator to notify the workers.
        When host-1 and host-3 are removed, driver.resume will call _activate_workers, which will update the host assignments.
        It has a check to see if the intersection between the previous and current set of hosts. It finds that the previous
        set is  ['host-1:2'], and the current set is ['host-2:1'], since there was no notification for the added and removed
        hosts.
        This ensures that the previous set of hosts can register before the current set is discovered.
        """
        if self.wait_for_previous_set is False:
            return
        while(self.prevlist and self.executor):
            for item in self.prevlist:
                host, slots = item.split(":")
                slot = self.executor.driver.get_slot_info(host, 0)
                # Avoid the empty slot
                if (not slot.hostname) or self.executor.driver.get_worker_client(slot):
                    break
            else:
                time.sleep(0.001)
                continue
            break
        self.prevlist = hostlist


class StatusCallback:
    def __init__(self):
        self._journal = []

    def __call__(self, info_dict):
        self._journal.append(info_dict)

    def fetch(self):
        return self._journal.copy()


def _create_training_function(iterations):
    def training_fn():
        import time
        import torch
        import horovod.torch as hvd
        from horovod.ray import ray_logger

        hvd.init()

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ray_logger.log({"started": True, "pid": os.getpid()})

        @hvd.elastic.run
        def train(state):
            for state.epoch in range(state.epoch, iterations):
                ray_logger.log({"training": True, "pid": os.getpid()})
                time.sleep(0.1)
                state.commit()  # triggers scale-up, scale-down
            ray_logger.log({"finished": True, "pid": os.getpid()})

        state = hvd.elastic.TorchState(
            model, optimizer, batch=0, epoch=0, commits=0, rendezvous=0)
        train(state)
        return True

    return training_fn


@contextmanager
def fault_tolerance_patches():
    with mock.patch(
            'horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS',
            0.1):
        with mock.patch(
                "horovod.runner.util.network.get_driver_ip",
                return_value=socket.gethostbyname(socket.gethostname())):
            yield


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_fault_tolerance_hosts_added_and_removed(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (10, ['host-1:2']),
            (30, ['host-1:2', 'host-2:1', 'host-3:1']),
            (None, ['host-2:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(min_num_proc=1, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)
        settings.discovery.executor = executor
        training_fn = _create_training_function(iterations=50)
        executor.start()
        trace = StatusCallback()
        results = executor.run(training_fn, callbacks=[trace])
        assert len(results) == 1

        events = trace.fetch()
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 1, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
@pytest.mark.skip(reason='https://github.com/horovod/horovod/issues/3197')
def test_fault_tolerance_hosts_remove_and_add(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (10, ['host-1:2', 'host-2:1', 'host-3:2']),
            (10, ['host-1:2']),
            (None, ['host-1:2', 'host-4:1', 'host-5:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(min_num_proc=1, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        training_fn = _create_training_function(iterations=30)
        executor.start()
        trace = StatusCallback()
        results = executor.run(training_fn, callbacks=[trace])
        assert len(results) == 4

        events = trace.fetch()
        assert sum(int("started" in e) for e in events) == 7, events
        assert sum(int("finished" in e) for e in events) == 4, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_max_num_proc(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (10, ['host-1:2']),
            (None, ['host-1:2', 'host-4:1', 'host-5:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_num_proc=1, max_num_proc=2, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        training_fn = _create_training_function(iterations=20)
        executor.start()
        trace = StatusCallback()
        results = executor.run(training_fn, callbacks=[trace])
        assert len(results) == 2

        events = trace.fetch()
        assert sum(int("started" in e) for e in events) == 2, events
        assert sum(int("finished" in e) for e in events) == 2, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_min_num_proc(ray_8_cpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (10, ['host-1:1']),
            (10, ['host-1:1', 'host-4:1', 'host-5:1']),
            (None, ['host-1:1', 'host-4:1', 'host-5:1', 'host-6:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_num_proc=4, max_num_proc=4, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, cpus_per_slot=1, override_discovery=False)

        training_fn = _create_training_function(iterations=30)
        executor.start()
        trace = StatusCallback()
        results = executor.run(training_fn, callbacks=[trace])
        assert len(results) == 4

        events = trace.fetch()
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 4, events


@pytest.mark.skipif(
    not gloo_built(), reason='Gloo is required for Ray integration')
def test_gpu_e2e(ray_8_cpus_gpus):
    with fault_tolerance_patches():
        discovery_schedule = [
            (10, ['host-1:1']),
            (10, ['host-1:1', 'host-4:1', 'host-5:1']),
            (None, ['host-1:1', 'host-4:1', 'host-5:1', 'host-6:1']),
        ]
        nics = list(psutil.net_if_addrs().keys())[0]

        settings = ElasticRayExecutor.create_settings(
            min_num_proc=4, max_num_proc=4, nics={nics})
        settings.discovery = SimpleTestDiscovery(discovery_schedule)
        executor = ElasticRayExecutor(
            settings, gpus_per_slot=1, use_gpu=True, override_discovery=False)

        training_fn = _create_training_function(iterations=30)
        executor.start()
        trace = StatusCallback()
        results = executor.run(training_fn, callbacks=[trace])
        assert len(results) == 4

        events = trace.fetch()
        assert sum(int("started" in e) for e in events) == 4, events
        assert sum(int("finished" in e) for e in events) == 4, events


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(sys.argv[1:] + ["-v", "-x", __file__]))
