# Refactoring Plan solving issue #66

## Step 0

- check that the tests are runnable

```shell
# Should specify in the doc that the git LFS extension is necessary
sudo apt install git-lfs
git lfs install
git lfs pull

# then install the dependencies
poetry install --with dev

# then run the tests
poetry run coverage run
```

## Step 1

- Apply the [Black](https://black.readthedocs.io/en/stable/) module on all the code (it correct inconsistent indentation and inconsistent use of quotes, making the code more readable)
- 
