# Contributing to Displacement Early Warning System

Thank you for your interest in contributing! This project aims to provide accessible, open-source tools for displacement risk analysis.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check if the issue already exists
2. Create a new issue with:
   - Clear description of the problem/suggestion
   - Steps to reproduce (for bugs)
   - Your R version and operating system
   - Example code or data (if applicable)

### Suggesting New Features

We welcome suggestions for:
- New data sources to integrate
- Additional ML models or validation approaches
- Improved visualizations
- Performance optimizations
- Documentation improvements

Open an issue with the "enhancement" label and describe:
- The feature and its use case
- How it would work
- Why it would be valuable

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following our coding standards (see below)
4. **Test your changes** thoroughly
5. **Commit with clear messages**: `git commit -m "Add feature: description"`
6. **Push to your fork**: `git push origin feature/your-feature-name`
7. **Submit a pull request** with a clear description

## Coding Standards

### R Code Style

Follow the [tidyverse style guide](https://style.tidyverse.org/):

```r
# Good
calculate_risk_score <- function(rent_change, demolitions) {
  risk <- 0.4 * rent_change + 0.3 * demolitions
  return(risk)
}

# Bad
CalculateRiskScore<-function(x,y){risk=0.4*x+0.3*y;return(risk)}
```

### Documentation

- **Functions**: Document with roxygen-style comments
- **Scripts**: Include header with purpose and usage
- **Complex code**: Add inline comments explaining logic

Example:
```r
#' Calculate spatial lag of a variable
#' 
#' @param data An sf object
#' @param var Variable name to calculate lag for
#' @param k Number of nearest neighbors
#' @return Vector of spatial lag values
calculate_spatial_lag <- function(data, var, k = 6) {
  # Implementation
}
```

### Commit Messages

Use clear, descriptive commit messages:

```
Good:
- "Add eviction data processing to 02_process_data.R"
- "Fix bug in spatial lag calculation"
- "Update README with new data source instructions"

Bad:
- "Update"
- "Fix stuff"
- "Changes"
```

## Priority Contribution Areas

We especially welcome contributions in these areas:

### 1. New Data Sources

Help integrate additional displacement indicators:
- Eviction filings
- Land value assessments  
- Corporate/investor ownership
- Building code violations
- Utility disconnections

See `data/README.md` for data format guidelines.

### 2. Model Improvements

Enhance prediction approaches:
- Time-series models for temporal dynamics
- Deep learning architectures
- Causal inference methods
- Model interpretability tools (SHAP, LIME)

### 3. Validation Methods

Improve validation approaches:
- Spatial cross-validation enhancements
- Temporal validation strategies
- Model calibration techniques
- Performance metrics for imbalanced data

### 4. Visualization

Create better visualizations:
- Dashboard frameworks (Shiny, flexdashboard)
- 3D visualizations
- Time-series animations
- Comparative city analyses

### 5. Documentation

Improve documentation:
- Case studies and examples
- Video tutorials
- Methodology explanations
- Academic references

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `source("packages.R")`
3. Set up Census API key
4. Run the full pipeline once to understand the workflow
5. Make your changes
6. Test thoroughly

## Testing

Before submitting a PR:

1. **Run the full pipeline** to ensure nothing breaks
2. **Check for errors** in each step
3. **Verify outputs** look reasonable
4. **Test edge cases** (missing data, extreme values)
5. **Document new features** in code and README

## Pull Request Process

1. **Update documentation** (README, code comments)
2. **Add your changes** to the appropriate section
3. **Test thoroughly** on your local machine
4. **Submit PR** with:
   - Clear title and description
   - Reference to related issue (if applicable)
   - Screenshots (for visual changes)
   - Performance notes (for optimization PRs)

We'll review PRs as quickly as possible and may request changes.

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Personal attacks
- Publishing others' private information

## Questions?

- **General questions**: Open a discussion
- **Bug reports**: Open an issue
- **Security issues**: Email maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Academic papers using this code (as appropriate)

## Getting Help

New to contributing? Check out:
- [First Timers Only](https://www.firsttimersonly.com/)
- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [GitHub's Guide to Pull Requests](https://docs.github.com/en/pull-requests)

Thank you for contributing to displacement prevention research! 🏘️
