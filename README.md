# VisualSLAM
Visual-based Navigation project

## Notes:
1. Please create a feature branch for your work and use master branch only for working solutions. 
2. **No build flies, images or executables are allowed** to be commited to the repository.
3. CMake is used to build the project

## How to build and run the code:
```bash
mkdir build
cmake ..
make
./slam ../data/left ../data/right 10

## only 22 for visualization otherwise we can only see pose data
./slam ../data/left/ ../data/right/ ../data/calib.txt 22
```


