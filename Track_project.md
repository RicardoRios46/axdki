# Day 01

## Progress
- Spend time understanding the logic to build A1. Too many dimentions!!
- Code may be good as it is structured right now. Need to test.
- Tried to build local dipy with UV (Failed)

## Todo
- Build local dipy with dipy offitial recommended build.
- Something is missing in the dipy build to link the new axdki module.
- Test the implementation after the first fit (A1).

# Day 02

## Progress
- A1 and A2 implementation are build into dipy
- Pixi enviroments are set up now for building dipu. Probably mamba is easier
- Module is integrated to meson bluid now


## ToDo
- Test and coument the implementation

# Day 03

## Progress
- Create notebook with testing implementaiton
- Moving the A1 and A2 creation to the fit method, so the model only receive gtab
- 

# Overall ToDo
- Documentation: Create an example page that document theory and fitting (dipy/docs/examples)
- Predict: Implement the foward model. Methods  Part of dipy/reconst/axdki. Used for pytest.
- Test: Implement pytest dipy/reconst/tests/tests_axdki

## Optional ToDo
- CLI: Should we implement a CLI interface???
- Implement regularization techinques as Hamilton 2024.
- Include the notebook implementation for the DKI