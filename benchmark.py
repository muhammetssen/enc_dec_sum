from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

args = PyTorchBenchmarkArguments(models=["google/mt5-small","google/mt5-base","facebook/mbart-large-cc25"], batch_sizes=[2, 4, 8],
                                 sequence_lengths=[128, 256, 512, 1024], speed=False)
benchmark = PyTorchBenchmark(args)
results = benchmark.run()
print(results)
