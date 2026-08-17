---
source_url: https://d2l.ai/chapter_computational-performance/index.html
title: 13. Computational Performance
chapter: '13'
section_number: '13'
date: null
extractor: d2l
source_type: web
---

# 13. Computational Performance

In deep learning, datasets and models are usually large, which involves
heavy computation. Therefore, computational performance matters a lot.
This chapter will focus on the major factors that affect computational
performance: imperative programming, symbolic programming, asynchronous
computing, automatic parallelism, and multi-GPU computation. By studying
this chapter, you may further improve computational performance of those
models implemented in the previous chapters, for example, by reducing
training time without affecting accuracy.

- 13.1. Compilers and Interpreters
 - 13.1.1. Symbolic Programming
 - 13.1.2. Hybrid Programming
 - 13.1.3. Hybridizing the Sequential Class
 - 13.1.4. Summary
 - 13.1.5. Exercises
- 13.2. Asynchronous Computation
 - 13.2.1. Asynchrony via Backend
 - 13.2.2. Barriers and Blockers
 - 13.2.3. Improving Computation
 - 13.2.4. Summary
 - 13.2.5. Exercises
- 13.3. Automatic Parallelism
 - 13.3.1. Parallel Computation on GPUs
 - 13.3.2. Parallel Computation and Communication
 - 13.3.3. Summary
 - 13.3.4. Exercises
- 13.4. Hardware
 - 13.4.1. Computers
 - 13.4.2. Memory
 - 13.4.3. Storage
 - 13.4.4. CPUs
 - 13.4.5. GPUs and other Accelerators
 - 13.4.6. Networks and Buses
 - 13.4.7. More Latency Numbers
 - 13.4.8. Summary
 - 13.4.9. Exercises
- 13.5. Training on Multiple GPUs
 - 13.5.1. Splitting the Problem
 - 13.5.2. Data Parallelism
 - 13.5.3. A Toy Network
 - 13.5.4. Data Synchronization
 - 13.5.5. Distributing Data
 - 13.5.6. Training
 - 13.5.7. Summary
 - 13.5.8. Exercises
- 13.6. Concise Implementation for Multiple GPUs
 - 13.6.1. A Toy Network
 - 13.6.2. Network Initialization
 - 13.6.3. Training
 - 13.6.4. Summary
 - 13.6.5. Exercises
- 13.7. Parameter Servers
 - 13.7.1. Data-Parallel Training
 - 13.7.2. Ring Synchronization
 - 13.7.3. Multi-Machine Training
 - 13.7.4. Key–Value Stores
 - 13.7.5. Summary
 - 13.7.6. Exercises
