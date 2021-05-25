if exist clean_repo.bat call cd ..
call conda activate moe
call nbdev_clean_nbs
call mkdocs build