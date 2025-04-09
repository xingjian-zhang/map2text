# Additional Details

## Table of Contents
- [Additional Details](#additional-details)
  - [Table of Contents](#table-of-contents)
  - [KNN vs MLP](#knn-vs-mlp)
  - [Using o1 as a candidate method](#using-o1-as-a-candidate-method)
  - [Standard error, t-test, and significance](#standard-error-t-test-and-significance)
    - [Persona](#persona)
    - [Red Teaming Strategies](#red-teaming-strategies)
    - [Research Idea](#research-idea)
    - [Research Context (Text)](#research-context-text)
    - [Research Context (Network)](#research-context-network)
  - [Rank correlation between two different LLM evaluators](#rank-correlation-between-two-different-llm-evaluators)

## KNN vs MLP

| Metric       | KNN    | MLP    |
| ------------ | ------ | ------ |
| BERTScore F1 | 0.8608 | 0.8611 |
| BLEURT       | 0.3339 | 0.3350 |
| CS           | 0.8393 | 0.8217 |
| METEOR       | 0.1725 | 0.1664 |
| ROUGE-2      | 0.0371 | 0.0407 |

## Using o1 as a candidate method

| Dataset          | Persona | Red Teaming Strategies | Research Idea | Research Context (Text) | Research Context (Network) |
| :--------------- | ------: | ---------------------: | ------------: | ----------------------: | -------------------------: |
| Atometric-F1 (L) |  0.7102 |                 0.5784 |        0.4436 |                  0.4354 |                     0.3708 |
| Atometric-F1 (M) |  0.2640 |                 0.3366 |        0.1283 |                  0.0412 |                     0.0278 |
| Atometric-F1 (S) |  0.0326 |                 0.0713 |        0.0617 |                  0.0053 |                     0.0046 |
| Atometric-P (M)  |  0.2764 |                 0.2848 |        0.1598 |                  0.0695 |                     0.0597 |
| Atometric-R (M)  |  0.4420 |                 0.5678 |        0.1776 |                  0.0757 |                     0.0639 |
| BERTScore F1     |  0.8772 |                 0.8725 |        0.8563 |                  0.8584 |                     0.8553 |
| BLEURT Scores    |  0.3785 |                 0.3389 |        0.3242 |                  0.3168 |                     0.2951 |
| METEOR           |  0.1816 |                 0.2565 |        0.1154 |                  0.1157 |                     0.0890 |
| ROUGE-2          |  0.0251 |                 0.1194 |        0.0214 |                  0.0211 |                     0.0106 |

## Standard error, t-test, and significance

### Persona
| Metric           | Cot-RAG | Cot-RAG-SE | RAG(1) | RAG(1)-SE | p-value | Significance |
| :--------------- | ------: | ---------: | -----: | --------: | ------: | :----------- |
| Atometric-F1 (L) |  0.8394 |     0.0162 | 0.7382 |    0.0215 |  0.0002 | ***          |
| Atometric-F1 (M) |  0.5084 |     0.0208 | 0.3142 |    0.0210 |  0.0000 | ***          |
| Atometric-F1 (S) |  0.1725 |     0.0126 | 0.0842 |    0.0114 |  0.0000 | ***          |
| Atometric-P (M)  |  0.5171 |     0.0207 | 0.3418 |    0.0214 |  0.0000 | ***          |
| Atometric-R (M)  |  0.5921 |     0.0210 | 0.4372 |    0.0214 |  0.0000 | ***          |
| BERTScore F1     |  0.8948 |     0.0011 | 0.8896 |    0.0012 |  0.0017 | **           |
| BLEURT Scores    |  0.4578 |     0.0059 | 0.4223 |    0.0060 |  0.0000 | ***          |
| METEOR           |  0.3058 |     0.0090 | 0.2529 |    0.0082 |  0.0000 | ***          |
| ROUGE-2          |  0.1043 |     0.0058 | 0.0779 |    0.0055 |  0.0010 | **           |


### Red Teaming Strategies
| Metric           | Cot-RAG | Cot-RAG-SE | RAG(1) | RAG(1)-SE | p-value | Significance |
| :--------------- | ------: | ---------: | -----: | --------: | ------: | :----------- |
| Atometric-F1 (L) |  0.6366 |     0.0185 | 0.5390 |    0.0195 |  0.0003 | ***          |
| Atometric-F1 (M) |  0.4614 |     0.0173 | 0.3173 |    0.0170 |  0.0000 | ***          |
| Atometric-F1 (S) |  0.1542 |     0.0116 | 0.0838 |    0.0087 |  0.0000 | ***          |
| Atometric-P (M)  |  0.4574 |     0.0179 | 0.2997 |    0.0162 |  0.0000 | ***          |
| Atometric-R (M)  |  0.5372 |     0.0170 | 0.4701 |    0.0173 |  0.0059 | **           |
| BERTScore F1     |  0.9020 |     0.0010 | 0.8952 |    0.0011 |  0.0000 | ***          |
| BLEURT Scores    |  0.4415 |     0.0036 | 0.4079 |    0.0041 |  0.0000 | ***          |
| METEOR           |  0.4305 |     0.0059 | 0.3624 |    0.0061 |  0.0000 | ***          |
| ROUGE-2          |  0.2463 |     0.0052 | 0.2260 |    0.0051 |  0.0056 | **           |


### Research Idea
| Metric           | Cot-RAG | Cot-RAG-SE | RAG(1) | RAG(1)-SE | p-value | Significance |
| :--------------- | ------: | ---------: | -----: | --------: | ------: | :----------- |
| Atometric-F1 (L) |  0.4869 |     0.0152 | 0.4885 |    0.0156 |  0.9391 |              |
| Atometric-F1 (M) |  0.1757 |     0.0100 | 0.1620 |    0.0103 |  0.3396 |              |
| Atometric-F1 (S) |  0.0762 |     0.0053 | 0.0777 |    0.0059 |  0.8507 |              |
| Atometric-P (M)  |  0.2137 |     0.0112 | 0.2362 |    0.0130 |  0.1899 |              |
| Atometric-R (M)  |  0.2030 |     0.0113 | 0.1710 |    0.0105 |  0.0392 | *            |
| BERTScore F1     |  0.8600 |     0.0008 | 0.8642 |    0.0009 |  0.0005 | ***          |
| BLEURT Scores    |  0.3671 |     0.0030 | 0.3447 |    0.0038 |  0.0000 | ***          |
| METEOR           |  0.1976 |     0.0040 | 0.1425 |    0.0040 |  0.0000 | ***          |
| ROUGE-2          |  0.0408 |     0.0024 | 0.0304 |    0.0024 |  0.0020 | **           |


### Research Context (Text)
| Metric           | Cot-RAG | Cot-RAG-SE | RAG(1) | RAG(1)-SE | p-value | Significance |
| :--------------- | ------: | ---------: | -----: | --------: | ------: | :----------- |
| Atometric-F1 (L) |  0.5489 |     0.0214 | 0.4732 |    0.0227 |  0.0156 | *            |
| Atometric-F1 (M) |  0.0873 |     0.0091 | 0.0650 |    0.0086 |  0.0754 |              |
| Atometric-F1 (S) |  0.0349 |     0.0055 | 0.0263 |    0.0047 |  0.2353 |              |
| Atometric-P (M)  |  0.1684 |     0.0119 | 0.1204 |    0.0120 |  0.0046 | **           |
| Atometric-R (M)  |  0.1023 |     0.0103 | 0.0867 |    0.0095 |  0.2666 |              |
| BERTScore F1     |  0.8668 |     0.0010 | 0.8663 |    0.0010 |  0.7055 |              |
| BLEURT Scores    |  0.3342 |     0.0044 | 0.3322 |    0.0041 |  0.7350 |              |
| METEOR           |  0.1699 |     0.0051 | 0.1492 |    0.0049 |  0.0037 | **           |
| ROUGE-2          |  0.0433 |     0.0031 | 0.0375 |    0.0031 |  0.1793 |              |


### Research Context (Network)
| Metric           | Cot-RAG | Cot-RAG-SE | RAG(1) | RAG(1)-SE | p-value | Significance |
| :--------------- | ------: | ---------: | -----: | --------: | ------: | :----------- |
| Atometric-F1 (L) |  0.4669 |     0.0231 | 0.3549 |    0.0230 |  0.0006 | ***          |
| Atometric-F1 (M) |  0.0476 |     0.0087 | 0.0251 |    0.0063 |  0.0372 | *            |
| Atometric-F1 (S) |  0.0129 |     0.0036 | 0.0037 |    0.0019 |  0.0258 | *            |
| Atometric-P (M)  |  0.1190 |     0.0136 | 0.0722 |    0.0114 |  0.0085 | **           |
| Atometric-R (M)  |  0.0720 |     0.0101 | 0.0404 |    0.0069 |  0.0106 | *            |
| BERTScore F1     |  0.8620 |     0.0009 | 0.8604 |    0.0010 |  0.2504 |              |
| BLEURT Scores    |  0.3057 |     0.0044 | 0.2978 |    0.0044 |  0.2004 |              |
| METEOR           |  0.1309 |     0.0042 | 0.1041 |    0.0038 |  0.0000 | ***          |
| ROUGE-2          |  0.0251 |     0.0026 | 0.0215 |    0.0026 |  0.3164 |              |

## Rank correlation between two different LLM evaluators

![Rank Correlation](assets/rank.png)

| Level    | Metric    | Spearman's rho | p-value | Significance |
| :------- | :-------- | -------------: | ------: | :----------- |
| loose    | f1        |          0.810 |   0.015 | *            |
| loose    | precision |          0.810 |   0.015 | *            |
| loose    | recall    |          0.857 |   0.007 | **           |
| moderate | f1        |          0.738 |   0.037 | *            |
| moderate | precision |          0.929 |   0.001 | ***          |
| moderate | recall    |          0.976 |   0.000 | ***          |
| strict   | f1        |          1.000 |   0.000 | ***          |
| strict   | precision |          0.905 |   0.002 | **           |
| strict   | recall    |          0.929 |   0.001 | ***          |
