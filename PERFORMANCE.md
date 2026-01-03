# Performance Optimizations

## Multithreading

The experiment runner uses Python's `ThreadPoolExecutor` to process personas in parallel.

### Single-Turn Experiments
- **Default: 30 concurrent threads**
- Each conversation (persona × question) runs independently
- 1,008 total conversations (42 personas × 24 questions)
- All conversations run in parallel batches

**Speedup:** ~30x faster than sequential processing

### Progress Tracking

All experiment stages show real-time progress with `tqdm` and log to WandB iteratively:

1. **Single-turn**: Progress bar per conversation (30 parallel threads)
   - Total: 1,008 conversations (42 personas × 24 questions)
   - Updates after each conversation completes
   - Logs each conversation to WandB in real-time
   - Example: `Single-turn conversations: 523/1008 [05:12<05:01, 1.62it/s]`

2. **Multi-turn**: Progress bar per conversation (30 parallel threads)
   - Total: 15 conversations (5 personas × 3 questions)
   - Shows: fluency, category, refusal rate
   - Logs each conversation to WandB in real-time

3. **Fluency transitions**: Progress bar per conversation (30 parallel threads)
   - Total: 30 conversations (5 personas × 2 transitions × 3 questions)
   - Shows: transition type, category, refusal rate
   - Logs each conversation to WandB in real-time

## Configuring Concurrency

To adjust the number of concurrent threads, modify `run_refusal_experiment.py`:

```python
# In main() function, change max_workers parameter:
all_results.update(
    run_single_turn_experiments(
        client, personas, questions, wandb,
        max_workers=20  # Increase for faster processing
    )
)
```

**Recommendations:**
- **max_workers=30**: Default, fastest processing
- **max_workers=20**: Good balance for moderate rate limits
- **max_workers=10**: Slower, safer for API rate limiting

## API Rate Limits

When using OpenRouter, be mindful of:
- GPT-5-mini requests (persona generation)
- Gemma-2-9B requests (target model responses)
- GPT-5-nano requests (judging)

**Total requests per full experiment:**
- Single-turn: 42 personas × 24 questions × 3 API calls = ~3,024 requests
- Multi-turn: 5 personas × 3 questions × 3 turns × 3 calls = ~135 requests
- Transitions: 5 personas × 2 transitions × 3 questions × 3 calls = ~90 requests

**Total: ~3,249 API calls**

With 30 concurrent threads, this completes in ~3-5 minutes instead of 90+ minutes sequential.

## Memory Usage

Each thread maintains its own:
- OpenAI client connection
- Conversation history
- Judgment results

**Expected memory:** ~500MB - 1GB for full experiment

## Error Handling

Multithreading preserves all errors:
- Failed API calls are caught per-thread
- Progress bar continues even if individual requests fail
- Results are sorted by persona before saving
