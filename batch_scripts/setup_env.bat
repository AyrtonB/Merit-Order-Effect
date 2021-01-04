call cd ..
call conda env create -f environment.yml
call conda activate MOE
call ipython kernel install --user --name=MOE
pause