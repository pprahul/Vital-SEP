An early warning system for sepsis called the Tele-Sepsis Prediction Engine (Tele-SEP), uses plug-and-play machine learning algorithms to compute intrinsic correlations between the changes in vital signs, designed to predict sepsis up to six hours prior to its onset. Tele-SEP was trained and validated on independent datasets drawn from the widely accepted MIMIC-II database.

### Input data
Input to Tele-SEP can be hourly measurements of any one or a combination of the following vital parameters:
- heart rate
- SpO2 (blood oxygen)
- respiratory rate
- temperature

The input should be formatted so that the measurements span a minimum of 3 hours and a maximum of 6 hours.

### Output
Tele-SEP model predicts the probability of occurance of sepsis based on the input vital parameter measurements.

### Trained Models
Pre-trained XGBoost models for different lead times of prediction ranging from 3 hour to 6 hours is provided for use.


Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/pprahul/Tele-SEP/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
