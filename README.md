# NanoFold

## Architecture

- Small transformer-like models (~50K parameters) for secondary structure prediction
- We use a "Stripped Puppy" architecture inspired by [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) and [Hyena-DNA](https://www.science.org/doi/10.1126/science.ado9336)
  - mixed attention with simplified hyena blocks as in stripped hyena models
  - recycling block for refining predictions with weight sharing for efficient parametrization

![image](stripped_hyena_drawing.png)