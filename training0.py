from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os, cv2, gc

X = []
y = []

#Carregar fotos
path ="./Saudaveis" #pega o caminho da pasta temporaria
photos = os.listdir(path) #pega todos arquivos que tem dentro dela
for i,file in enumerate(photos):
    frame = cv2.imread(path+'/'+file)
    resized_frame = cv2.resize(frame, (100, 50))
    frameArray = resized_frame.flatten()
    X.append(frameArray)
    y.append(0)
    #limpar da memoria
    if (i % 20 == 0):
        gc.collect()
    print(i+1)
    # if(i > 10):
    #     break
print(5*"\n*")
print("\nACABOU DE CARREGAR AS IMAGENS SAUDAVEIS\n")
print(5*"*\n")

#Carregar fotos
path ="./Doentes" #pegando o caminho da pasta
photos = os.listdir(path) #pega todos arquivos que tem dentro dela
for i,file in enumerate(photos):
    frame = cv2.imread(path+'/'+file)
    resized_frame = cv2.resize(frame, (100, 50))
    frameArray = resized_frame.flatten()
    X.append(frameArray)
    y.append(1)
    #limpar memoria
    if (i % 20 == 0):
        gc.collect()
    print(i+1)
    # if(i > 10):
    #     break
print(5*"\n*")
print("\nACABOU DE CARREGAR AS IMAGENS DOENTES\n")
print(5*"*\n")

X = pd.DataFrame(X)

gc.collect()
print(5*"\n*")
print("\nACABOU DE CARREGAR o DataFrame\n")
print(5*"*\n")
#passando pelo PCA

pca = PCA(0.99).fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)

#Separando teste para treino
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

print('\n*****************RESULTADOS****************************\n\n')

# Criando rede RandonForest
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.15)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
