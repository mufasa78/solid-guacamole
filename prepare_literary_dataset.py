import pandas as pd
import os

# Create output directory if it doesn't exist
os.makedirs('prepared_datasets', exist_ok=True)

# Read the crosslingual_literary dataset
print("Reading crosslingual_literary.csv...")
df = pd.read_csv('datasets/crosslingual_literary.csv')

print(f"Original shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Filter only Chinese and English texts
print("Filtering Chinese and English texts...")
df = df[df['language'].isin(['zh', 'en'])]

# Map emotions to integers
print("Mapping emotions to integers...")
emotion_mapping = {emotion: i for i, emotion in enumerate(sorted(df['emotion'].unique()))}
df['label'] = df['emotion'].map(emotion_mapping)

# Save the emotion mapping
emotion_mapping_df = pd.DataFrame({
    'emotion': list(emotion_mapping.keys()),
    'label': list(emotion_mapping.values())
})
emotion_mapping_df.to_csv('prepared_datasets/emotion_mapping.csv', index=False)
print("Emotion mapping:")
print(emotion_mapping_df)

# Select only the necessary columns
df = df[['text', 'label']]

# Save the prepared dataset
print("Saving prepared dataset...")
df.to_csv('prepared_datasets/prepared_literary.csv', index=False)

print(f"Final shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Now let's create a sample expanded_novels.csv file for training
print("\nCreating sample expanded_novels.csv file...")

# Create a sample dataset with 8 classes (genres)
genres = ["玄幻", "武侠", "都市", "言情", "科幻", "历史", "游戏", "悬疑"]
texts = [
    # 玄幻 (Fantasy)
    "少年得到了一本古老的功法秘籍，开始了自己的修仙之路，他将穿越各种奇幻世界，寻找更强大的力量。",
    "在这个世界上，有着各种各样的魔法生物，主角意外获得了一种罕见的魔法能力，开始了自己的冒险之旅。",
    "修真界风云变幻，各大门派争夺天地灵宝。少年从山村走出，凭借祖传玉佩中蕴含的上古仙人传承，一步步踏上修仙之路。",
    
    # 武侠 (Martial Arts)
    "剑客独立山巅，长剑出鞘，剑气纵横三千里。他的武功已达到了出神入化的境界，整个江湖都在传颂他的名号。",
    "少年拜入名门，学习绝世武功，为父报仇，行侠仗义，打遍天下无敌手。",
    "江湖风波起，刀光剑影中，少侠凭借一身绝世轻功和精湛剑法，解救了无数被欺压的百姓，成为了侠义的象征。",
    
    # 都市 (Urban)
    "年轻的白领在公司里勤勤恳恳工作，却遭到同事的排挤和上司的刁难，他决定辞职创业，开始了自己的奋斗史。",
    "大学毕业后，主角来到大城市打拼，面对高昂的房价和激烈的职场竞争，他努力适应都市生活，寻找自己的位置。",
    "都市精英白天是成功的投资经理，晚上却有着不为人知的另一面，他游走在城市的明与暗之间，寻找生活的真谛。",
    
    # 言情 (Romance)
    "女主角在一次偶然的机会下认识了英俊多金的男主角，两人经历了一系列误会和波折后，最终走到了一起。",
    "青梅竹马的两人从小一起长大，却在成年后因为各种原因分开，多年后再次相遇，发现彼此的感情从未改变。",
    "事业有成的女强人遇到了温柔体贴的男子，打破了她对爱情的防备，两人携手共同面对生活中的各种挑战。",
    
    # 科幻 (Sci-Fi)
    "宇宙飞船穿越虫洞，来到了一个全新的星系。这里的科技水平远超地球，人类正在与外星文明建立第一次接触。",
    "在不久的将来，人工智能已经高度发达，主角发现了一个可能威胁人类存亡的AI阴谋，他必须阻止这场灾难。",
    "地球资源枯竭，人类开始向宇宙深处探索，寻找新的家园。一支探险队在遥远的星球上发现了神秘的外星遗迹。",
    
    # 历史 (Historical)
    "年轻的将军在乱世中崛起，带领军队征战四方，最终统一了分裂的国家，建立了一个强大的帝国。",
    "落魄书生通过科举考试一举成名，入朝为官，在复杂的官场中周旋，既要实现自己的抱负，又要保全性命。",
    "乱世之中，主角从一个普通的农家子弟成长为叱咤风云的一代枭雄，他的一生跨越了整个王朝的兴衰。",
    
    # 游戏 (Gaming)
    "主角意外进入了一个虚拟游戏世界，在这里，他必须不断升级自己的能力，完成各种任务，才能找到回家的路。",
    "职业电竞选手为了梦想不断努力，经历了无数失败和挫折，最终带领团队赢得了世界冠军。",
    "游戏公司的程序员发现自己开发的游戏中出现了奇怪的BUG，玩家在游戏中的行为开始影响现实世界。",
    
    # 悬疑 (Mystery)
    "连环杀人案震惊全城，年轻的刑警凭借敏锐的洞察力，一步步接近真相，却发现案件背后隐藏着更大的阴谋。",
    "古老宅邸中接连发生离奇死亡事件，主角作为继承人来到这里，却发现家族中流传着一个可怕的诅咒。",
    "知名侦探接受委托调查一起看似简单的失踪案，随着调查深入，他发现这起案件牵涉到一个庞大的犯罪组织。"
]

# Create a DataFrame
expanded_novels_df = pd.DataFrame({
    'text': texts,
    'label': [i // 3 for i in range(len(texts))]  # 3 examples per class
})

# Save the dataset
expanded_novels_df.to_csv('datasets/expanded_novels.csv', index=False)
print(f"Created expanded_novels.csv with {len(expanded_novels_df)} samples")
print(f"Class distribution:\n{expanded_novels_df['label'].value_counts()}")

print("\nDataset preparation complete!")
