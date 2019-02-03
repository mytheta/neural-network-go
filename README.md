## 概要
golangでニューラルネットワーク

## やりたいこと
XOR問題をニューラルネットワークで解く．  
入力層3つ，中間層3つ，出力層1つにし，class1なら1,class2なら0の教師ラベルを設定する．  
![sample](https://github.com/mytheta/neural-network-go/blob/master/neural.png)



##　学習データ
class1=(0,0),(1,1)  
class2=(1.0),(0,1)  
のニクラス分類を行う.  
![sample](https://github.com/mytheta/neural-network-go/blob/master/report.png)

## 活性化関数
中間層，出力層それぞれの活性化関数にシグモイド関数を用いた．

## 重みの更新
誤差逆伝播法でそれぞれのユニットの誤差を求めて，重みの更新を行う．

## 誤差関数
![sample](https://github.com/mytheta/neural-network-go/blob/master/points.png)
