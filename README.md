# ICQF
Implementation of Interpretability Constrained Questionnaire Factorization

### Prerequisites
The `environment.yml` records all libraries used to reproduce the environment.

### Remark
Given that both HBN and ABCD dataset require the acceptance of user agreement, we are not allowed to share both raw and pre-processed data directly.
To reproduce the results reported in the manuscript, please visit https://healthybrainnetwork.org/ and https://abcdstudy.org/ for more information.

Having said that, the preprocessing demonstration on CBCL-HBN questionnaire can be found in `/data/HBN/HBN_loaders/CBCL.ipynb`

For readability, the evaluation are implemented as jupyter notebook for illustration purpose.
### Getting start
In the following, we use HBN dataset as an example for instruction.

1. Obtain the following questionnaire information (filename may vary, depends on data version)
  - basic demographic information (`./data/HBN/9994_Basic_Demos_20210310.csv`)
  - subject's diagnostic label (`./data/HBN/HBN_participantDx.csv`)
  - question information (`./data/HBN/CBCL2001_item_variables_fixed.csv`)
  - question responses (`./data/HBN/9994_CBCL_20210310.csv`)

2. Process the information as demonstrated in `./data/HBN/HBN_loaders/CBCL.ipynb`.

3. Run `CBCL-HBN_facroize.ipynb` for both model training and model inference

4. Evaluate the model with `CBCL-HBN_evaluate.ipynb`
