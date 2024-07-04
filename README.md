# Daily Paper in 3 Sentences
**Keywords**: Dataset, Education, LLM, Korean, Vocab Expansion, Paraphrase Generation, Knowledge Distillation

[2024.07.04]
### Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models
Archive 2024, <https://arxiv.org/pdf/2402.14714>
_Korean | LLM | Vocab Expansion_
+ English-centric LLM의 tokenizer를 사용했을 때 한국어는 더 많은 토큰으로 분리됨 -> longer response times, shorter context lengths, higher API costs
+ 새로운 vocab이 반영된 embedding layer와 LM head를 업데이트하기 위해 6단계의 순차적 훈련 -> 모델이 기존 token과 새로운 token들을 align할 수 있도록
+ pretraining 단계에서 Transformer layer들을 업데이트하기 위해 QLoRA 활용, fine-tuning 단계에서는  DPO 적용 및 영어 intruction tuning 데이터 번역해서 활용

---------------------------------------
# + $\alpha$
[2024.07.03]   
### RACE: Large-scale Reading Comprehension Dataset From Examinations   
EMNLP 2017, <https://arxiv.org/pdf/1704.04683v5>   
_Dataset | Education_
+ 12-18세 학생을 대상으로 하는 중고등학교 영어 시험 문제들로부터 데이터 수집
+ question과 answer candidate들이 passage의 span이 아닌 경우가 많음
+ Reasoning difficutly 5단계에 따라 질문들을 분류함

[2024.07.04]
### Parameter Efficient Diverse Paraphrase Generation Using Sequence-Level Knowledge Distillation
ICACS 2024, <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10473289>   
_Paraphrase Generation, LLM, Knowledge Distillation_
+ ChatGPT를 활용한 data-centric sequence-level knowledge distillation (T5-small, Flan T5-small, BART-base 등 small LM에 LoRA까지 사용하여 computationally efficient하게 훈련)
+ ChatGPT의 temperature를 0으로 설정한 뒤 주어진 sentence의 paraphrase들을 list 형식으로 반환하게함 -> 200만 쌍의 paraphrase pair를 구축
+ Inference 단계에서 decoding hyperparameter들을 유연하게 조절 (top-p, top-k, beam size 등)
