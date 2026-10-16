"""Check that physical batching preserves completion-only token weighting."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


def test_padded_batch_loss_and_gradients_match_token_weighted_examples():
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, attention_dropout=0, pad_token_id=0))
    model.config.use_cache = False
    model.eval()
    rows = [
        {'input_ids': [1, 2, 3, 4], 'completion_mask': [0, 0, 1, 1]},
        {'input_ids': [1, 5, 6], 'completion_mask': [0, 0, 1]},
    ]
    collator = DataCollatorForLanguageModeling(pad_token_id=0, completion_only_loss=True)
    batch = collator(rows)
    assert batch['attention_mask'].sum().item() == 7
    assert (batch['labels'][:, 1:] != -100).sum().item() == 3
    assert batch['labels'][1, -1].item() == -100
    batched_loss = model(**batch).loss
    batched_loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    expected = sum(model(**collator([row])).loss * targets / 3
                   for row, targets in zip(rows, [2, 1]))
    torch.testing.assert_close(batched_loss, expected, rtol=1e-5, atol=1e-6)
    expected.backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=1e-4, atol=1e-6)
