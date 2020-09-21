###  create a new repository on the command line

echo "# satk" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin https://github.com/vito19960522/satk.git
git push -u origin master
                

### push an existing repository from the command line

git remote add origin https://github.com/vito19960522/satk.git
git branch -M master
git push -u origin master