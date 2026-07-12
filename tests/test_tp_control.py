import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lite_llama"
    / "executor"
    / "tp_control.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "lite_llama_tp_control_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TensorCommandCodecTest(unittest.TestCase):
    def test_prefill_round_trip(self):
        module = load_module()
        requests = [
            SimpleNamespace(
                control_id=4,
                prompt_tokens=[10, 11],
                max_new_tokens=32,
                temperature=0.6,
                top_p=0.9,
            ),
            SimpleNamespace(
                control_id=9,
                prompt_tokens=[20],
                max_new_tokens=16,
                temperature=0.0,
                top_p=1.0,
            ),
        ]

        encoded = module.encode_prefill(requests)
        decoded = module.decode_command(*encoded)

        self.assertEqual(decoded.operation, "prefill")
        self.assertEqual(decoded.control_ids, [4, 9])
        self.assertEqual(decoded.prompt_tokens, [[10, 11], [20]])
        self.assertEqual(decoded.max_new_tokens, [32, 16])
        self.assertEqual(decoded.min_tokens, [0, 0])
        self.assertEqual(decoded.temperatures, [0.6, 0.0])
        self.assertEqual(decoded.top_ps, [0.9, 1.0])


    def test_store_payload_round_trip_uses_cpu_serialization(self):
        module = load_module()
        requests = [
            SimpleNamespace(
                control_id=1,
                prompt_tokens=[101, 102, 103],
                max_new_tokens=8,
                temperature=0.0,
                top_p=1.0,
            )
        ]

        payload = module.encode_store_payload(module.encode_prefill(requests))
        decoded = module.decode_store_payload(payload)

        self.assertEqual(decoded.operation, "prefill")
        self.assertEqual(decoded.control_ids, [1])
        self.assertEqual(decoded.prompt_tokens, [[101, 102, 103]])
        self.assertEqual(decoded.max_new_tokens, [8])
        self.assertEqual(decoded.min_tokens, [0])
        self.assertEqual(decoded.temperatures, [0.0])
        self.assertEqual(decoded.top_ps, [1.0])

    def test_store_command_channel_does_not_allocate_device_tensors(self):
        module = load_module()
        source = MODULE_PATH.read_text(encoding="utf-8")
        start = source.index("class StoreCommandChannel")
        end = source.index("class TensorCommandChannel")
        store_source = source[start:end]

        self.assertNotIn("torch.tensor", store_source)
        self.assertNotIn("torch.empty", store_source)
        self.assertNotIn("torch.distributed.broadcast", store_source)


    def test_store_command_channel_send_receive_with_cpu_store(self):
        module = load_module()

        class FakeStore:
            def __init__(self):
                self.values = {}

            def set(self, key, value):
                self.values[key] = value

            def get(self, key):
                return self.values[key]

        store = FakeStore()
        sender = module.StoreCommandChannel(store=store)
        receiver = module.StoreCommandChannel(store=store)
        sequence = sender.send(module.encode_decode([7, 8]))

        received_sequence, decoded = receiver.receive_with_sequence()

        self.assertEqual(sequence, 0)
        self.assertEqual(received_sequence, 0)
        self.assertEqual(decoded.operation, "decode")
        self.assertEqual(decoded.control_ids, [7, 8])

    def test_store_command_channel_has_ack_protocol(self):
        source = MODULE_PATH.read_text(encoding="utf-8")
        start = source.index("class StoreCommandChannel")
        end = source.index("class TensorCommandChannel")
        store_source = source[start:end]

        self.assertIn("def ack", store_source)
        self.assertIn("def wait_ack", store_source)
        self.assertIn("/ack/", store_source)

    def test_decode_release_and_shutdown_round_trip(self):
        module = load_module()

        for operation, encoder in (
            ("decode", module.encode_decode),
            ("release", module.encode_release),
        ):
            decoded = module.decode_command(*encoder([3, 8]))
            self.assertEqual(decoded.operation, operation)
            self.assertEqual(decoded.control_ids, [3, 8])

        shutdown = module.decode_command(*module.encode_shutdown())
        self.assertEqual(shutdown.operation, "shutdown")
        self.assertEqual(shutdown.control_ids, [])

    def test_decode_state_round_trip(self):
        module = load_module()
        requests = [
            SimpleNamespace(
                control_id=3,
                model_context_tokens=[10, 11, 12],
            ),
            SimpleNamespace(
                control_id=8,
                model_context_tokens=[20, 21],
            ),
        ]

        decoded = module.decode_command(*module.encode_decode_state(requests))

        self.assertEqual(decoded.operation, "decode_state")
        self.assertEqual(decoded.control_ids, [3, 8])
        self.assertEqual(decoded.expected_seq_lens, [3, 2])
        self.assertEqual(decoded.prompt_tokens, [])

    def test_prefill_chunk_round_trip(self):
        module = load_module()
        requests = [
            SimpleNamespace(
                control_id=4,
                prompt_tokens=[10, 11, 12],
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
                prefill_cursor=2,
            )
        ]

        decoded = module.decode_command(*module.encode_prefill_chunk(requests, 64))

        self.assertEqual(decoded.operation, "prefill_chunk")
        self.assertEqual(decoded.control_ids, [4])
        self.assertEqual(decoded.prompt_tokens, [[10, 11, 12]])
        self.assertEqual(decoded.max_new_tokens, [32])
        self.assertEqual(decoded.min_tokens, [0])
        self.assertEqual(decoded.temperatures, [0.0])
        self.assertEqual(decoded.top_ps, [1.0])
        self.assertEqual(decoded.prefill_cursors, [2])
        self.assertEqual(decoded.chunk_size, 64)


if __name__ == "__main__":
    unittest.main()
