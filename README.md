# SC24-Long_Exposure-AD

We propose **Long Exposure**, an efficient system for accelerating the parameter-efficient fine-tuning for LLMs. 

We implement **Long Exposure** as an end-to-end fine-tuning system, which makes three main contributions, to summarize: 

1. $C_1$ We are the first to identify and leverage the intrinsic sparsity within LLM fine-tuning, namely shadowy sparsity, to accelerate the PEFT process for LLMs.
2. $C_2$ We introduce three key components that capture, predict, and exploit sparsity patterns, respectively. This approach provides a coherent strategy for optimizing both the multi-head attention and the MLP block within LLMs.
3. $C_3$ We implement these techniques as an end-to-end fine-tuning system that is compatible with a variety of PEFT techniques. Our system achieves up to 2.49× speedups and 2.77× memory savings compared to the state-of-arts without accuracy degradation.
