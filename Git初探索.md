# 常用基本命令

## create a new repository on the command line

echo "# satk" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin <https://github.com/vito19960522/satk.git>
git push -u origin master

## push an existing repository from the command line

git remote add origin <https://github.com/vito19960522/satk.git>
git branch -M master
git push -u origin master

## 同步远程分支的删减变化到本地

git remote prune origin

## 分支管理

git branch -a
git branch -d old_branch
git checkout -b new_branch

## 远程配置

git remote add origin <https://github.com/vito19960522/satk.>
git remote remove origin
