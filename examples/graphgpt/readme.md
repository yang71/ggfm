# GraphGPT

- Paper link: [GraphGPT: Graph Instruction Tuning for Large Language Models](https://arxiv.org/abs/2310.13023)
- Author's code repo: [https://github.com/HKUDS/GraphGPT](https://github.com/HKUDS/GraphGPT)

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
    <td>  /</td>
    <td>  /</td>
    <td>0.5136</td>
    <td>0.2678</td>
  </tr>
</table>

Beacause the original code is not available
(it's only make graph data to node classification data and train the model with node classification task)
, we cannot reproduce the results of link prediction task. We only provide the results of node classification task.
