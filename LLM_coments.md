1~3 paragraphs of comments including questions, weakness, or new ideas

[2024.09.12]
### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)
JMLR 2020, <https://arxiv.org/pdf/1910.10683>   
+ A unified framework that converts all text-based language problems into a text-to-text format.
+ pre-training causes the model to develop general-purpose abilities and knowledge.
+ Compare the effectiveness of different transfer learing objectives, unlabeled datasets, and other factors.
+ T5는 Transformer의 original architecture에서 layer norm과 position embedding scheme이 조금 수정됨
+ 한 모델을 다양한 task로 훈련 -> prefix를 통해 task 구분
+ causal language modeling보다 denoising objective (masked language modeling)이 더 좋은 성능을 보인다는 연구결과를 따름

Question: English pretraining data만 활용했다는데, translation task는 어떻게?? 다른 언어에 대한 vocab의 embedding이 어떻게 있지? -> vocab 구축할 때 번역 데이터의 언어까지 포함시켜서 구축함
Masking된 consecutive token sequence만을 output으로 예측하는 objective function은 완전한 sentence를 generation하는 downstream task과 align되는 것 같지 않아보였지만, 오히려 downstream task에서는 비슷하거나 더 높은 성능을 보였음. 대신 이 방식이 훨씬 efficient하기 때문에 computational cost를 고려한다면 더 좋은 방안이 됨.


T5 is a unified framework that converts all text-based language problems into a text-to-text format. The goal of this paper is **transferring** general-purpose abilities and knowledge obtained **from** pre-training stage to the downstream tasks (**with** fine-tuning). There **is** no significant differences between the architecture of the original Transformer and T5 (except for the details **of** layer norm and position embedding). For the translation downstream task, the authors **involves** the vocabulary of specific languages (used in the translation tasks) **to** the model's embedding layers. I wonder **that the addition of** vocabs **in** non-English languages before or after pre-training affects to the model performance in **the downstream task**. And the pre-training objective used for T5 is really effective in downstream tasks that require fluent natural language generation (such as response generation or story generation). The benchmark downstream datasets require the model to generate relatively short and simple output than the tasks mentioned above.

[REVISED]   
T5 is a unified framework that converts all text-based language problems into a text-to-text format. The goal of this paper is **to transfer the** general-purpose abilities and knowledge obtained **during the** pre-training stage to downstream tasks **through** fine-tuning. There **are** no significant differences between the architecture of the original Transformer and T5, except for details **related to** layer normalization and position embeddings. For the translation downstream task, the authors **incorporate** the vocabulary of specific languages (used in translation tasks) **into** the model's embedding layers. I wonder **whether adding** vocabularies **of** non-English languages before or after pre-training affects the model's performance in **downstream tasks**. Additionally, I question whether the pre-training objective used for T5 is truly effective for downstream tasks that require fluent natural language generation, such as response generation or story generation. The benchmark datasets for downstream tasks typically require the model to generate relatively short and simple outputs compared to the more complex tasks mentioned above.

[2024.09.18]
### MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases
ICML 2024, <https://arxiv.org/pdf/2402.14905>   
+ A mobile app should not exceed 10% of the DRAM (usually 6GB~12GB), since DRAM is shared with the OS and other apps.
+ an iPhone can support conversational use an entire day by utilizing 350M 8-bit model consuming only 0.035J/token.
+ Evaluate the pre-trained model on zero-shot common sense reasoning tasks. (small-size 모델이라면
+ Model architecture의 component들을 하나씩 바꿔가며 성능 측정? (Empirical approach 활용?)
+ LLM에서는 parameter 개수, 훈련 데이터 크기, iteration 횟수 등이 중요하지만 작은 모델에서는 architecture 자체가 중요함.
+ Experiments with 19 small models consistently demonstrate that deeper and thinner models outperform their shallower and wider counterparts.

Question: small model들에 비해 성능이 많이 높아졌다고 하더라도, LLM zero-shot 성능보다 매우 떨어짐 -> zero-shot으로의 활용성이 있을지 의문임. small-size model의 장점은 fine-tuning이 간편하다는 것인데, 여기서는 Chat과 API-calling task 두가지에 대해서만 fine-tuning을 진행함. 과연 이전까지의 높은 성능을 보였단 encoder기반 혹은 encoder-decoder 모델들보다 fine-tuning 성능도 높을지 의문임
![image](https://github.com/user-attachments/assets/ec272530-4203-4947-9c77-c471cce21fca)
