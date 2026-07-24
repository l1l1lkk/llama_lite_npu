import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "lite_llama" / "tracing.py"
)
SPEC = importlib.util.spec_from_file_location(
    "lite_llama_tracing_test", MODULE_PATH
)
TRACING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACING
SPEC.loader.exec_module(TRACING)

ModelLayerTracer = TRACING.ModelLayerTracer
TraceLifecycleObserver = TRACING.TraceLifecycleObserver
TraceManager = TRACING.TraceManager
TracingBackend = TRACING.TracingBackend
current_batch_context = TRACING.current_batch_context
rank_output_path = TRACING.rank_output_path


class FakeClock:
    def __init__(self, value=100):
        self.value = value

    def __call__(self):
        self.value += 1
        return self.value


class FakeHandle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class FakeTensor:
    shape = (2, 4, 16)


class FakeLayer:
    def __init__(self):
        self.before = None
        self.after = None
        self.handles = []

    def register_forward_pre_hook(self, callback):
        self.before = callback
        handle = FakeHandle()
        self.handles.append(handle)
        return handle

    def register_forward_hook(self, callback):
        self.after = callback
        handle = FakeHandle()
        self.handles.append(handle)
        return handle

    def run(self):
        inputs = (FakeTensor(),)
        self.before(self, inputs)
        output = FakeTensor()
        self.after(self, inputs, output)
        return output


class FakeLayerContainer:
    def __init__(self, layers):
        self.layers = layers

    def named_children(self):
        return [(str(index), layer) for index, layer in enumerate(self.layers)]


class FakeModel:
    def __init__(self, layers):
        self.container = FakeLayerContainer(layers)

    def named_modules(self):
        return [
            ("", self),
            ("model.layers", self.container),
        ]


class FakeBackend:
    def __init__(self, layer=None):
        self.layer = layer
        self.seen_context = None

    def decode(self, requests):
        self.seen_context = current_batch_context()
        if self.layer is not None:
            self.layer.run()
        return [7 for _ in requests]

    def prefill(self, requests):
        self.seen_context = current_batch_context()
        return [8 for _ in requests]

    def prefill_chunk(self, requests, chunk_size):
        self.seen_context = current_batch_context()
        return [None for _ in requests]

    def release(self, requests):
        return None

    def preempt(self, requests):
        return None


def fake_request(request_id="request-1"):
    return SimpleNamespace(
        request_id=request_id,
        control_id=3,
        endpoint="chat",
        prompt_tokens=[1, 2, 3],
        generated_token_ids=[],
        max_new_tokens=8,
        prefill_cursor=0,
        preemptions=0,
        finish_reason=None,
    )


class TraceManagerTests(unittest.TestCase):
    def test_level_filter_and_bounded_ring(self):
        manager = TraceManager(
            enabled=True,
            level="request",
            max_events=3,
            clock_ns=FakeClock(),
            wall_clock_ns=FakeClock(1_000),
        )
        self.assertIsNone(
            manager.emit("scheduler_state", event_level="scheduler")
        )
        manager.emit("one")
        manager.emit("two")
        manager.emit("three")
        snapshot = manager.snapshot(limit=10)
        self.assertEqual(
            [event["event"] for event in snapshot["events"]],
            ["one", "two", "three"],
        )
        self.assertEqual(snapshot["dropped_events"], 1)
        self.assertEqual(snapshot["newest_seq"], 4)

    def test_jsonl_writer_flushes_on_close(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "trace.jsonl"
            manager = TraceManager(
                enabled=True,
                level="scheduler",
                output_path=output,
            )
            manager.emit(
                "scheduler_state",
                event_level="scheduler",
                waiting=2,
                prefilling=1,
                running=3,
            )
            manager.close()
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(records[0]["event"], "trace_started")
        self.assertEqual(records[-1]["waiting"], 2)

    def test_lifecycle_events_are_redacted(self):
        manager = TraceManager(enabled=True, level="request")
        observer = TraceLifecycleObserver(manager)
        request = fake_request()
        request.prompt_text = "must not be recorded"
        observer.on_request_submitted(request)
        request.generated_token_ids.append(5)
        observer.on_token(request)
        request.finish_reason = "stop"
        observer.on_request_finished(request)
        records = manager.snapshot(limit=20)["events"]
        encoded = json.dumps(records)
        self.assertNotIn("must not be recorded", encoded)
        self.assertEqual(records[-1]["generated_tokens"], 1)
        self.assertEqual(records[-1]["finish_reason"], "stop")

    def test_backend_context_correlates_layer_events(self):
        manager = TraceManager(enabled=True, level="layer", rank=1)
        layer = FakeLayer()
        tracer = ModelLayerTracer(FakeModel([layer]), manager)
        self.assertEqual(tracer.install(), 1)
        backend = FakeBackend(layer)
        wrapped = TracingBackend(backend, manager, rank=1)
        request = fake_request()
        self.assertEqual(wrapped.decode([request]), [7])
        self.assertIsNotNone(backend.seen_context)
        self.assertEqual(backend.seen_context.request_ids, ("request-1",))

        events = manager.snapshot(limit=20)["events"]
        names = [event["event"] for event in events]
        self.assertEqual(
            names[-4:],
            ["decode_start", "layer_start", "layer_end", "decode_end"],
        )
        batch_ids = {
            event["batch_id"] for event in events if "batch_id" in event
        }
        self.assertEqual(len(batch_ids), 1)
        self.assertEqual(events[-2]["layer_index"], 0)
        self.assertEqual(events[-2]["rank"], 1)

        tracer.close()
        self.assertTrue(all(handle.removed for handle in layer.handles))

    def test_tensor_parallel_output_path_is_rank_scoped(self):
        path = rank_output_path(
            "traces/session.jsonl", rank=1, tensor_parallel=True
        )
        self.assertEqual(path.name, "session.rank1.jsonl")
        self.assertEqual(
            rank_output_path(
                "traces/session.jsonl", rank=1, tensor_parallel=False
            ).name,
            "session.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
