# DeZero 学習用ミニ実装

本レポジトリは書籍『ゼロから作るDeep Learning』の前半ステージを題材に、`Variable`・`Function` を中心とした自動微分フレームワークを段階的に実装した学習用コードです。

## 使い方
```bash
python main.py
```
`y.data` と `x.grad` が出力され、合成関数 `square(exp(square(x)))` の順伝播と逆伝播の結果が確認できます。

## テスト
```bash
python -m unittest tests/*.py  
```