# Music Genre Classification Project Report
##### `NE6081014` `陳冠友`

## 程式執行環境
- `Ubuntu 18.04 LTS`
- `Python3`
- 使用套件
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `librosa`

## 使用的 audio features
利用以下 features 組成每首音樂的 feature vector
- `Chroma Frequencies`
  - 將 spectrum 投射到 12 個 bin，每個 bin 分別表示 12 個半音
- `Spectral Centroid`
- `Spectral Rolloff`
- `Spectral Bandwidth`
- `Zero Crossing Rate`
- `Tempogram`
- `mfcc`

### Feature vector 產生
利用 `librosa` 套件中的 feacture extraction functions 分別對每個檔案取出上述的 audio features，並將回傳的 2-D array 取平均轉成 scalar 後當作 feature value。而 `mfcc` 因為有 20 個 channel，因此回傳的 2-D array 取平均後仍然是 2-D。所以 feature vector 中共有 20 個欄位是 `mfcc` 中 20 個 channel 取平均後的值。

## 分類的方法與模型
### KNeighbor Classifier
> Given a test instance i, find the k closest neighbors and their labels
Predict i’s label as the majority of the labels of the k nearest neighbors  

利用 KNN 演算法分別對資料進行 10 分類，並透過 5-fold cross validation 計算平均的 training 與 validation accuray。`K` 值嘗試過使用 `3` `5` 或 `10`，最後發現當 `K=5` 時有最好的 valid accuracy，因此以下數據為 `K=5` 的模型數據。  
|  | average accuracy | max accuracy | min accuracy |
| :---: | :---: | :---: | :---: |
| Train | 0.7705 | 0.7925 | 0.7425 |
| Valid | 0.675 | 0.73 | 0.63 |

#### Remove zero corssing rate feature
上課的投影片有提到 spectral 或 rhythm based 的 feature 對 music genre classification 可能有較好的表現，因此試著移除 zero crossing rate 這個 feature 再訓練一次 KNN classifier，並同樣將 `K` 設為 `5`
|  | average accuracy | max accuracy | min accuracy |
| :---: | :---: | :---: | :---: |
| Train | 0.772 | 0.797 | 0.755 |
| Valid | 0.684 | 0.73 | 0.62 |

可以發現移除 zero crossing rate 後有稍微提升了 accuracy，但不明顯。因此接下來將採用 SVM 進行分類。
### SVM classifier
利用 SVM 進行 10 分類，同樣移除 zero crossing rate 並使用 5-fold cross validation。  
|  | average accuracy | max accuracy | min accuracy |
| :---: | :---: | :---: | :---: |
| Train | 0.844 | 0.865 | 0.8325 |
| Valid | 0.71 | 0.77 | 0.62 |
發現 SVM 的分類表現比 KNN 更為凸出。
## Future Work
這次的資料集數量比較小，所以不太適合使用 deep learning 的 model 進行分類，但若資料集數量夠大的話，也許可以使用 `RNN` 或 `Transformer` encoder 對 audio file 進行 encode 再利用其 hidden representation 進行分類。