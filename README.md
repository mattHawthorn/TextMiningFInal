# TextMiningFinal

Questions:
- Is it beneficial to cluster documents in topic space vs. clustering them in word space?
- What's the best clustering technique in the topic case?

Data:
- 20 Newsgroups: https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups
    - 20 classes
    - 20000 instances
- Reuters-21578 https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection
    - 135 'topics'
    - 540 'categories' - places, people, orgs, exchanges
    - 21578 instances
- Metrics:
    - Normalized mutual information
    - Mean F-score across hierarchical levels
    - F-score @ true number of topics
