import torch
import pickle
import numpy as np
from preprocess import preprocess_text, tokenize_and_pad
from utils import load_model

def classify_text(text, model_name="Deep_CNN_Model"):
    """
    使用训练好的模型对文本进行分类
    
    参数:
    text: 要分类的文本
    model_name: 模型名称
    
    返回:
    预测的类别和概率
    """
    # 加载模型和词汇表
    model, vocab = load_model(model_name)
    model.eval()
    
    # 加载处理后的数据信息
    with open('deep_processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    max_len = processed_data['max_len']
    class_names = processed_data['class_names']
    
    # 预处理文本
    processed_text = preprocess_text(text)
    
    # 转换为序列并填充
    sequence = tokenize_and_pad(processed_text, vocab, max_len)
    
    # 转换为tensor
    input_tensor = torch.tensor(sequence).unsqueeze(0)  # 添加批次维度
    
    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # 获取预测类别和概率
    pred_class = predicted.item()
    pred_prob = probs[0][pred_class].item()
    
    # 获取所有类别的概率
    all_probs = probs[0].numpy()
    class_probs = {class_names[i]: float(all_probs[i]) for i in range(len(class_names))}
    
    return {
        'predicted_class': class_names[pred_class],
        'confidence': pred_prob,
        'class_probabilities': class_probs
    }

# 测试样例
if __name__ == "__main__":
    # 玄幻示例
    fantasy_text = "少年握紧手中的古老玉佩，感受到一股神秘的力量在体内流转。他知道，自己的修仙之路才刚刚开始，前方还有无数的妖魔鬼怪和强大的敌人等待着他。"
    
    # 武侠示例
    martial_arts_text = "老者的剑法如行云流水，每一招都蕴含着数十年的修为。年轻人虽然天赋异禀，但在这位江湖隐士面前，仍然显得稚嫩无比。他知道，要想成为真正的剑客，还需要更多的磨练。"
    
    # 都市示例
    urban_text = "张明看着公司的季度报表，额头上的汗珠滑落。如果这次企划案不能通过，他可能面临被裁员的风险。就在这时，他的手机响了，是一个陌生的号码，对方自称可以帮他解决所有问题，但条件是..."
    
    # 言情示例
    romance_text = "林小雨和陈远是从小一起长大的青梅竹马，但大学毕业后，两人因为工作分隔两地。五年后的同学聚会上，当林小雨再次看到陈远时，那份埋藏在心底的感情又重新涌了上来。"
    
    # 科幻示例
    scifi_text = "太空站的警报声响起，宇航员陈刚迅速检查各项数据。一个不明物体正在接近，它的行为模式不符合任何已知的航天器。当通讯系统突然接收到一段奇怪的信号时，陈刚意识到，人类可能不是宇宙中唯一的智慧生物。"
    
    # 历史示例
    history_text = "大将军站在城墙上，望着远处敌军的营帐。三天后就是决战之日，这一战将决定两国的命运。作为皇帝最信任的将领，他肩负着保家卫国的重任，但朝中暗流涌动，他不知道背后有多少双眼睛在盯着他的一举一动。"
    
    # 游戏示例
    game_text = "李浩戴上VR头盔，登录了《永恒之境》。作为公会的会长，他需要带领团队攻克最新的副本。这个副本难度极高，全球只有三个团队成功通关，而今天，他们将挑战第四个通关记录。"
    
    # 悬疑示例
    mystery_text = "警探李明站在案发现场，这是一个密室，门窗都从内部锁住，没有任何强行闯入的痕迹。死者躺在地上，面部表情扭曲，似乎在死前经历了极度的恐惧。更奇怪的是，房间内找不到任何可能的凶器。"
    
    # 测试所有模型
    models = ["Deep_CNN_Model", "Deep_RNN_Model", "Deep_Transformer_Model"]
    
    test_texts = {
        "玄幻": fantasy_text,
        "武侠": martial_arts_text,
        "都市": urban_text,
        "言情": romance_text,
        "科幻": scifi_text,
        "历史": history_text,
        "游戏": game_text,
        "悬疑": mystery_text
    }
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        for genre, text in test_texts.items():
            print(f"\nTesting {genre} text:")
            print(f"Text: {text[:100]}...")
            
            try:
                result = classify_text(text, model_name)
                print(f"Predicted: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.4f}")
                
                # 打印所有类别的概率
                print("Class probabilities:")
                for cls, prob in sorted(result['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  {cls}: {prob:.4f}")
                
                # 检查是否正确
                if result['predicted_class'] == genre:
                    print("✓ Correct prediction!")
                else:
                    print("✗ Incorrect prediction!")
            except Exception as e:
                print(f"Error: {str(e)}")
