
# Instalar librerías de python
pip install -r requirements.txt
pip install --proxy http://xxxxxx -r requirements.txt

# generar archivo requirements.txt
pip freeze > requirements.txt


# GIT
## subir archivos al repositorio git
git status --> info
git add . -->
git commit -m "texto"
git push

## bajar archivos del repositorio git
git pull 

## cuando hay conflictos
git checkout . --> deshace los últimos cambios en local

