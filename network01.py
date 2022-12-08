#ニューラルネットワーククラスの定義
class neuralNetwork:

	#ニューラルネットワークの初期化
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		#入力層，隠れ層，出力層のノード数の設定
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		
		#学習率の設定
		self.lr = learningrate
		pass
		
	#ニューラルネットワークの学習
	def train():
		pass
		
	#ニューラルネットワークへの照会
	def query():
		pass
		
#入力層，隠れ層，出力層のノード数
input_nodes = 5
hidden_nodes = 5
output_nodes = 5

#学習率 = 0.5
learning_rate = 0.5

#ニューラルネットワークのインスタンスの生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
