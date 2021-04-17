from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

args = PyTorchBenchmarkArguments(models=["google/mt5-base","facebook/mbart-large-cc25","dbmdz/bert-base-turkish-uncased","bert-base-multilingual-uncased"], batch_sizes=[8, 16, 32,64],
                                 sequence_lengths=[32,64,256,512], speed=False)
benchmark = PyTorchBenchmark(args)
results = benchmark.run()
print(results)
