from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os, cv2, gc

X = []
y = []

classes = pd.read_excel('classes.xlsx')

#Carregar fotos
path ="./Photos" #pega o caminho da pasta
photos = os.listdir(path) #pega todos arquivos que tem dentro dela
for i, photo in enumerate(photos):
    frame = cv2.imread(path+'/'+photo)
    resized_frame = cv2.resize(frame, (100, 50))
    frameArray = resized_frame.flatten()#transforma a "matriz" em vetor
    X.append(frameArray)
    classe = classes.loc[classes['File'] == photo]['Binary.Label'].to_list()[0]#encontrar a foto no mapeamento de classes
    y.append(classe)
    #limpar da memoria
    if (i % 20 == 0):
        gc.collect()
    print(i+1)

print(5*"\n*")
print("\nACABOU DE CARREGAR AS IMAGENS\n")
print(5*"*\n")

y = pd.DataFrame(y)

y.replace('healthy', 0, inplace=True)
y.replace('unhealthy', 1, inplace=True)

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

print(2*"\n*")
print("\nResultado com PCA\n")
print(2*"*\n")

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.15)

clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
