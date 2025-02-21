# HiGPT

- Paper link: [HiGPT: Heterogeneous Graph Language Model](https://doi.org/10.48550/arXiv.2402.16024)
- Author's code repo: [https://github.com/HKUDS/HiGPT](https://github.com/HKUDS/HiGPT)

# Dataset Statics

Refer to [OAG-CS](https://ggl.readthedocs.io/en/latest/api/ggl.datasets.html#ggl.datasets.OAG-CS).

Results
-------

<table>
  <tr>
    <th>Task</th>
    <th colspan="2">Link Prediction</th>
    <th colspan="2">Node Classification</th>
  </tr>
  <tr>
    <th>Evaluation Matrix</th>
    <th>MRR</th>
    <th>NDCG</th>
    <th>Micro-F1</th>
    <th>Macro-F1</th>
  </tr>
  <tr>
    <td>Result</td>
    <td>-</td>
    <td>-</td>
    <td>0.5689</td>
    <td>0.2950</td>
  </tr>
</table>

Note on Link Prediction: The original HiGPT paper did not demonstrate its capabilities in link prediction tasks. Additionally, our tests have shown that the model is not well-suited for classification tasks involving thousands of categories, making it impractical for link prediction on the OAG-CS dataset.