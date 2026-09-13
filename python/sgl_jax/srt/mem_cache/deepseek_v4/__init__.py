"""DeepSeek V4 cache resources: storage, compressor state and host allocation.

The scheduler owns allocation; the model returns immutable pool updates. Global
request slots and rank-local original-token addresses retain the C1 contract.
"""
