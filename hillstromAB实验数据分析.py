import numpy as np
import pandas as pd
import statsmodels.stats.proportion as prop
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from pandas import crosstab
from scipy import stats
from patsy import build_design_matrices

#导入数据
df = pd.read_csv("hillstromAB.csv")

# 生成每个用户的编号
df["source_row_id"] = range(1, len(df) + 1)  # 第一条数据记录编号为 1



################################################################
# 一、整体实验效果检查
################################################################



######### 1.检查数据基本结构、缺失值、字段类型。没有缺失值。
#print(df)
print(f"数据前15行如下：\n{df.head(15).to_string()}\n")

print(f"数据中的缺失值分布：\n{df.isna().sum()}\n")

check1_0 = ["mens", "womens", "newbie", "visit", "conversion"]
if not df[check1_0].isin([0,1]).all().all():
    print("数据中有0、1之外的值")

#print(df.info())

# 查看各实验分组人数，女装主题邮件组：21387人，男装主题邮件组：21307人，不接收邮件组：21306人。
print(f"各实验分组人数如下：\n{df["segment"].value_counts()}\n")

# 计算三组汇总表格，包括该组的用户数、访问过的用户数、访问率、购买过的用户数、购买转化率、该组用户人均消费额
summary = df.groupby("segment").agg(
    n_customers = ("conversion", "size"),
    n_visitors = ("visit", "sum"),
    visit_rate = ("visit", "mean"),
    n_buyers = ("conversion", "sum"),
    conversion_rate = ("conversion", "mean"),
    spend_per_customer = ("spend", "mean"),
)
print(f"汇总表格如下：\n{summary.to_string()}\n")



######### 2.两个邮件版本分别与未发邮件组比较（与不接收邮件组比较，男装主题邮件组增加了多少人均销售额？女装主题邮件组增加了多少人均销售额？）

# 对照组、男装主题邮件组、女装主题邮件组的消费额数据
control_spend = df.loc[df["segment"]=="No E-Mail", "spend"]
men_spend = df.loc[df["segment"]=="Mens E-Mail", "spend"]
women_spend = df.loc[df["segment"]=="Womens E-Mail", "spend"]

# Welch 两独立样本t检验如下。结论：
# 在实验后两周内，未发邮件组的人均销售额是0.6528，男装版邮件组是1.4226，女装版邮件组是1.0772。
# 与未发邮件相比，男装版邮件带来的人均增量销售额估计为0.7698 （95%CI：0.4851 - 1.0545），p = 1.164 * 10^(-7)，
# 与未发邮件相比，女装版邮件带来的人均增量销售额估计为0.4244 （95%CI：0.1690 - 0.6799），p = 0.001129。
# 两个置信区间都高于0，表明两个实验都提高了实验期间的人均消费额。

test_men_con = stats.ttest_ind(men_spend, control_spend, equal_var=False)
ci_men_con = test_men_con.confidence_interval(confidence_level=0.95)
increase_men_con = men_spend.mean() - control_spend.mean()
print(
    f"男装主题邮件组人均增量：{increase_men_con:.4f}\n",
    f"男装主题邮件组p值：{test_men_con.pvalue:.4g}\n",
    f"男装主题邮件组 95% 置信区间：[{ci_men_con.low:.4f}, {ci_men_con.high:.4f}]\n",sep=''
)

test_women_con = stats.ttest_ind(women_spend, control_spend, equal_var=False)
ci_women_con = test_women_con.confidence_interval(confidence_level=0.95)
increase_women_con = women_spend.mean() - control_spend.mean()
print(
    f"女装主题邮件组人均增量：{increase_women_con:.4f}\n",
    f"女装主题邮件组p值：{test_women_con.pvalue:.4g}\n",
    f"女装主题邮件组 95% 置信区间：[{ci_women_con.low:.4f}, {ci_women_con.high:.4f}]\n",sep=''
)



######### 3.两个实验组互相比较（与女装主题邮件组相比，男装主题邮件组人均销售额提升了多少？）
# Welch 两独立样本t检验如下。结论：
# 与女装主题邮件相比，男装主题邮件带来的人均增量销售额估计为0.3454 （95%CI：0.0326 - 0.6583），p = 0.03047。
# 置信区间高于0，表明总体来看，对于提升人均销售额来说，男装主题邮件优于女装主题邮件。

test_men_women = stats.ttest_ind(men_spend, women_spend, equal_var=False)
ci_men_women = test_men_women.confidence_interval(confidence_level=0.95)
increase_men_women = men_spend.mean() - women_spend.mean()
print(
    f"男装主题邮件组相较于女装主题邮件组的人均销售额增量：{increase_men_women:.4f}\n",
    f"两实验组比较的p值：{test_men_women.pvalue:.4g}\n",
    f"男装主题邮件组相较于女装主题邮件组的人均销售额增量的 95% 置信区间：[{ci_men_women.low:.4f}, {ci_men_women.high:.4f}]\n",sep=''
)



######### 4.用访问率和购买率来衡量两实验组是否有差异：
n_mens = int(summary.loc["Mens E-Mail", "n_customers"])
n_womens = int(summary.loc["Womens E-Mail", "n_customers"])

n_vis_mens = int(summary.loc["Mens E-Mail", "n_visitors"])
n_vis_womens = int(summary.loc["Womens E-Mail", "n_visitors"])

n_buy_mens = int(summary.loc["Mens E-Mail", "n_buyers"])
n_buy_womens = int(summary.loc["Womens E-Mail", "n_buyers"])

# 双侧两比例z检验如下。结论：
# 换用访问率和购买率来衡量两实验组是否有差异，结果中两组排名没有反转。男装版的访问率、购买率的点估计更高。访问率、购买率差异的置信区间不包含0，
# 表明总体来看，对于提升访问率、购买率来说，男装主题邮件优于女装主题邮件。
z_vis, p_vis = prop.proportions_ztest(
    count=[n_vis_mens, n_vis_womens],
    nobs=[n_mens, n_womens],
    alternative="two-sided",
)

lower_vis, upper_vis = prop.confint_proportions_2indep(
    n_vis_mens, n_mens, n_vis_womens, n_womens,
    compare="diff",
    method="wald",
)
print(
    f"男装主题邮件组与女装主题邮件组比较访问率，访问率差：{(n_vis_mens/n_mens - n_vis_womens/n_womens)*100:.3f}%\n",
    f"男装主题邮件组与女装主题邮件组比较访问率，p值：{p_vis:.4g}\n",
    f"男装主题邮件组与女装主题邮件组比较访问率，95% 置信区间：[{lower_vis*100:.3f}%, {upper_vis*100:.3f}%]\n",sep=''
)

z_buy, p_buy = prop.proportions_ztest(
    count=[n_buy_mens, n_buy_womens],
    nobs=[n_mens,n_womens],
    alternative="two-sided",
)

lower_buy, upper_buy = prop.confint_proportions_2indep(
    n_buy_mens, n_mens, n_buy_womens, n_womens,
    compare="diff",
    method="wald",
)

print(
    f"男装主题邮件组与女装主题邮件组比较购买率，购买率差：{(n_buy_mens/n_mens - n_buy_womens/n_womens)*100:.3f}\n",
    f"男装主题邮件组与女装主题邮件组比较购买率，p值：{p_buy:.4g}\n",
    f"男装主题邮件组与女装主题邮件组比较购买率，95% 置信区间：[{lower_buy*100:.3f}%, {upper_buy*100:.3f}%]\n",sep=''
)




################################################################
# 二、客群差异分析
################################################################



######### 1.在不同客群中，男装版与女装版邮件组的表现是否有差异？这种差异在各客群组之间有不同吗？



# 创建客群标签
print(df["recency"].describe()) #检查recency最小、最大值

data = df.loc[df["segment"].isin(["Mens E-Mail", "Womens E-Mail"])] # 获取男装版与女装版邮件两个实验组数据
data["recency_group"] = pd.cut(
    data["recency"],
    bins=[-float("inf"), 3, 6, 9, float("inf")],
    labels=["1-3个月内","4-6个月","7-9个月","10-12个月"],
) # 给recency分组，3个月一组

data["mens_email"] = data["segment"].eq("Mens E-Mail").astype(int) # 创建实验组标签，1=男装版邮件组，0=女装版

purchase_labels = {
    (1,0):"只买过男装",
    (0,1):"只买过女装",
    (1,1):"男女装都买过",
    (0,0):"男女装都没买过",
}

data["purchase_group"] = [
    purchase_labels[mens,womens]
    for mens, womens in zip(data["mens"], data["womens"])
] # 创建历史购买男装、女装组

print(data.head().to_string())
print(crosstab(data["recency_group"], data["segment"]), end="\n\n")
print(crosstab(data["purchase_group"], data["segment"]))


# 回答第一个问题：在每个客群中，男装版与女装版邮件组的表现是否有差异？
# 把数据按各个客群变量分类，对于每个客群变量内部的每个客群，提取其中男装版邮件组和女装版邮件组的数据，进行两独立样本t检验
# 输出每个客群变量内部的每个客群的男装版邮件组人均销售护额 - 女装版邮件组人均销售护额，以及p值和置信区间
group_cols = [
    "purchase_group",   # 历史购买品类
    "newbie",           # 新老客
    "channel",          # 历史购买渠道
    "zip_code",         # 地区类别
    "recency_group",    # 距上次购买时间
    "history_segment",  # 历史消费档位
]

for group_col in group_cols:
    print(f"\n===== 按 {group_col} 分群 =====")

    for label, group in data.groupby(group_col, observed=True):
        mens = group.loc[group["mens_email"] == 1, "spend"]
        womens = group.loc[group["mens_email"] == 0, "spend"]

        print(f"\n{label}（男装版 {len(mens)} 人，女装版 {len(womens)} 人）")

        if len(mens) < 2 or len(womens) < 2:
            print("至少一个邮件组少于 2 人，跳过检验。")
            continue

        diff = mens.mean() - womens.mean()
        test = stats.ttest_ind(mens, womens, equal_var=False)
        ci = test.confidence_interval(confidence_level=0.95)

        print(f"男装版 − 女装版：{diff:+.4f} 美元/人\n")
        print(f"95% CI：[{ci.low:.4f}, {ci.high:.4f}]，p = {test.pvalue:.4g}")


# 回答第二个问题，每个客群内部 男装版与女装版邮件组的表现的差异，在该客群变量的各客群之间是否有差异？

# 用“最近购买月份分组”、“历史消费分组”、“曾经购买过的款式”、“是否是新人”、“渠道”、“zip码地区”构建回归方程
# 回归方程中邮件主题组别与各个客群分组项的交叉项的系数，就是各客群变量内部，每个客群的男装邮件人均销售额-女装邮件人均销售额 与参考客群该值的差值。
# 参考客群是：只买过男装、不是新人、web渠道、zipcode乡村地区、上次购买1-3个月内、历史消费金额0-100$之间。
# 再对模型交互项 以客群变量为为组进行检验，得到各客群变量内部各客群 男装邮件人均销售额 - 女装邮件人均销售额 是否相同 的p值与置信区间。

model = smf.ols(formula="""
spend ~ mens_email * (
    C(purchase_group, Treatment(reference="只买过男装")) +
    C(newbie, Treatment(reference = 0)) +
    C(channel, Treatment(reference = "Web")) +
    C(zip_code, Treatment(reference = "Rural")) +
    C(recency_group, Treatment(reference = "1-3个月内")) +
    C(history_segment, Treatment(reference = "1) $0 - $100"))
)
""",
                data=data, missing="raise").fit(cov_type="HC3", use_t=False)
print(model.summary())

tests = model.wald_test_terms(combine_terms=[":"], scalar=True).table
interaction_tests = tests.loc[tests.index.str.contains(":", regex=False)]
interaction_tests = interaction_tests.rename(index={":": "全部主题邮件组的交互项"})

print(f"\n交互项检验：\n{interaction_tests.to_string()}")

# 结果总结：
# 以邮件发出后两周的人均消费额为指标，对 42694 名收到男装版或女装版邮件的客户进行分析。
# 组内 Welch 检验显示，在 22 项客群比较中，有 6 项的未校正p值低于 0.05，方向均为男装版的人均消费额较高。
# 其中，过去同时买过男装和女装的客户，男装版比女装版高 1.6631 美元/人（95% CI 0.3315 至 2.9947）。
# 历史消费 200–350 美元和 350–500 美元的客户，差值分别为 0.8597 美元/人（95% CI 0.2430 至 1.4764）和 1.6698 美元/人（95% CI 0.4835 至 2.8561）。
# 另外三项出现在老客、郊区客户和距上次购买 1–3 个月的客户中。
# 若将这 22 项比较作为同一检验族进行 Holm 校正，没有一项达到 0.05，因此这些结果应视为探索性线索。

# 将六类客户特征及其与邮件版本的交互项同时纳入回归后，使用 HC3 稳健标准误得到的整体 Wald 检验显示，数据支持两版邮件的人均消费差随客户特征变化（p=0.00735）。
# 但逐个特征的交互项联合检验均未达到 0.05，其中历史购买品类的 p=0.0617，历史消费档位的 p=0.1078。
# 因此，现有结果提供了整体差异的证据，但尚未明确识别出可用于选择邮件版本的单一特征。
# 过去两类服装都买过的客户，以及历史消费在 200–500 美元的客户，可以作为下一轮分群实验优先验证的对象。



######### 2.根据实验结果，应该分别向哪些客群投放男装版和女装版邮件？哪些数据支持这些建议？
# 男装版邮件
# 建议在下一轮分群实验中，优先验证过去同时买过男装和女装的客户，以及历史消费在 200–350 美元和 350–500 美元的两个客群。
# 这三个客群中，男装版比女装版的人均消费额分别高 1.66、0.86 和 1.67 美元。但这些是探索性结果，22 项组内比较经 Holm 校正后均未达到 0.05。
# 交互回归虽提供了整体差异的证据（p=0.00735），各客户特征的交互项联合检验均未达到 0.05。因此，这些客群适合优先复验，目前还不能据此固定投放规则。

# 女装版邮件
# 暂无足够证据推荐某个客群优先接收女装版。
# 例如，在过去只买过女装的客户中，男装版减女装版的人均消费差仅为 −0.016 美元，95% 置信区间为 −0.425 至 0.393 美元。
# 建议在下一轮实验中继续保留女装版作为比较对象。




################################################################
# 三、投放决策
################################################################



######### 如果只能向最值得投放的10000名客户投放邮件，应该选择哪些客户，为什么？如果要排除10000名最不值得投放的呢？



# 1. 准备候选客户和实验数据
customers = df.copy().reset_index(drop=True)

# 创建购买类型历史标签变量
customers["purchase_group"] = [
    purchase_labels[(mens, womens)]
    for mens, womens in zip(customers["mens"], customers["womens"])
]



# 1. 保留三个邮件主题实验组，按 70% / 30% 划分训练集与验证集。

train = customers.sample(frac=0.70, random_state=191699)
valid = customers.drop(index=train.index).copy()

# 2. 只用 train 训练集拟合交互回归，根据用户的各个特征，包括购买类型历史、是否为新人、购买渠道、地区、距上次购买月份、历史消费金额分组，预测花费。
# 只有加入交互项，才能使不同的人有不同的 男装版预测 - 不发邮件预测（男装版增量预测）与女装版增量预测。
# 如果没有交互项，所有人的男装版增量预测相同，女装版增量预测也相同。无法根据每个人的特征推荐出合适的邮件主题，也没办法给每个人的预期增量排序。

model = smf.ols(formula = """spend ~ C(segment) * (
    C(purchase_group) + 
    C(newbie) + 
    C(channel) + 
    C(zip_code) + 
    recency + 
    C(history_segment)
)
""",
                data=train, missing="raise").fit()

# 3. 对原本的同一批客户预测三次，每次只改变邮件版本
scores = customers.copy()

pred_none = model.predict(customers.assign(segment="No E-Mail"))
pred_mens = model.predict(customers.assign(segment="Mens E-Mail"))
pred_womens = model.predict(customers.assign(segment="Womens E-Mail"))

scores["lift_mens"] = pred_mens - pred_none
scores["lift_womens"] = pred_womens - pred_none

scores["best_email"] = np.where(scores["lift_mens"] >= scores["lift_womens"], "Mens E-Mail", "Womens E-Mail")

scores["best_uplift"] = scores[["lift_mens", "lift_womens"]].max(axis=1)

# 4. 取出valid 验证集客户的预测分数，按相同投放比例，选择用户进行模拟邮件投放，未被选择的用户一律模拟预测为“不投放”。
valid = scores.loc[valid.index].sample(frac=1, random_state=191699).copy()

# 全体 64000 人发 10000 封，等于验证集也选约 15.625%
n_valid_send = round(len(valid) * 10_000 / len(customers))
selected = valid.nlargest(n_valid_send, "best_uplift")

valid["recommended_action"] = "No E-Mail"
valid.loc[selected.index, "recommended_action"] = selected["best_email"]

# 5. 用验证集里的真实实验结果估计策略价值
# 两种策略作比较：1.对于验证集中 预期销量增量前 15.625% 的用户投放最优邮件，其余用户不投放邮件；
#              2.对于验证集中的所有用户均不投放邮件。
# 两种策略下计算出的人均销售额相减，就是在验证集中 采用投放策略带来的人均销售额增益。
# 在选择投放的用户中，模拟分配为投放男装主题邮件的用户，只有约 1/3 的概率 其在初始A/B实验中被真实地分配为了男装主题。
# 因此，用 模拟分配为投放男装主题邮件的用户 中 真实分配结果是男装邮件的用户 计算出来的人均消费，仅为预期模拟效果的 1/3。需乘 3 才是预期模拟效果。
# 对于被模拟分配为女装邮件、不发邮件的用户来说同理，都要将 用 真实分配结果与模拟分配结果相匹配的数据计算的人均消费 乘 3 才是预期模拟效果。


matched = valid["segment"] == valid["recommended_action"]
no_email = valid["segment"] == "No E-Mail"

policy_value = (valid["spend"] * matched * 3).mean()
no_email_value = (valid["spend"] * no_email * 3).mean()

print("验证集入选客户的推荐版本：")
print(selected["best_email"].value_counts())
print(f"\n个性化投放：{policy_value:.4f} 美元/候选客户")
print(f"全部不发：{no_email_value:.4f} 美元/候选客户")
print(f"估计增量：{policy_value - no_email_value:+.4f} 美元/候选客户")

# 6. 在全体客户中生成模拟名单
# 先打乱，让分数相同的客户按可复现的随机顺序入选
scores = scores.sample(frac=1, random_state=191699)
scores = scores.sort_values(by="best_uplift", ascending=False)

scores.insert(0, "rank", range(1, len(scores) + 1)) # 生成排名列

send_to = scores.head(10000)
exclude = scores.tail(10000)

print("\n前 10000 人的推荐版本：")
print(send_to["best_email"].value_counts())

print(f"\n前 10000 人的id标签（前15行）：\n{send_to.head(15).to_string()}")
print(f"\n末尾 10000 人的id标签（前15行）：\n{exclude.head(15).to_string()}")


# 计算投放策略带来的人均销售额增益的置信区间
gain = 3 * valid["spend"] * (matched.astype(int) - no_email.astype(int))

estimate = gain.mean()
se = gain.std(ddof=1) / len(gain) ** 0.5

print(f"\n估计增量：{estimate:+.4f} 美元/候选客户")
print(f"近似 95% CI：[{estimate - 1.96 * se:.4f}, {estimate + 1.96 * se:.4f}]")


# 结果：发现投放策略带来的人均销售额增益、置信区间受随机种子选择影响较大。
# 当种子为13579时，估计增量：+0.2498 美元/候选客户，近似 95% CI：[0.0269, 0.4728]，
# 当种子为191699时，估计增量：-0.0773 美元/候选客户，近似 95% CI：[-0.2681, 0.1136]。
# 个性化策略的验证增量对训练集—验证集划分较为敏感，收益尚未得到稳定验证。
# 推测可能原因是数据中大额消费比较多。当一个大额消费 600$ 被分配到验证集中前 15% 同时与模型推荐匹配时，该消费对增量的贡献是：
# 3 * 600 / 19200 = 0.09375$.
# 同样，一个大额消费在不同随机种子中被分到验证集或训练集中，也会影响训练模型参数。


from pathlib import Path
from statsmodels.stats.multitest import multipletests

# 只比较男装版和女装版
group_data = data.loc[
    data["segment"].isin(["Mens E-Mail", "Womens E-Mail"])
].copy()

group_names = {
    "purchase_group": "历史购买品类",
    "newbie": "新老客",
    "channel": "历史购买渠道",
    "zip_code": "地区类别",
    "recency_group": "距上次购买时间",
    "history_segment": "历史消费档位",
}

# 来自你之前第 5 题的六类特征、两版邮件交互回归
interaction_p = {
    "purchase_group": 0.061690,
    "newbie": 0.820574,
    "channel": 0.357716,
    "zip_code": 0.262668,
    "recency_group": 0.278903,
    "history_segment": 0.107797,
}

rows = []

for field, title in group_names.items():
    for label, group in group_data.groupby(field, observed=True):
        mens = group.loc[group["segment"] == "Mens E-Mail", "spend"]
        womens = group.loc[group["segment"] == "Womens E-Mail", "spend"]

        test = stats.ttest_ind(mens, womens, equal_var=False)
        ci = test.confidence_interval()

        display_label = str(label)
        if field == "newbie":
            display_label = "新客" if label == 1 else "老客"

        rows.append({
            "group_type": title,
            "group_label": display_label,
            "n_mens": len(mens),
            "n_womens": len(womens),
            "mean_mens": mens.mean(),
            "mean_womens": womens.mean(),
            "diff": mens.mean() - womens.mean(),
            "ci_lower": ci.low,
            "ci_upper": ci.high,
            "p_raw": test.pvalue,
            "interaction_p": interaction_p[field],
        })

subgroup_results = pd.DataFrame(rows)

# 将全部 22 项组内比较作为同一检验族校正
subgroup_results["p_holm"] = multipletests(
    subgroup_results["p_raw"],
    method="holm",
)[1]

output_path = Path("subgroup_results.csv")
subgroup_results.to_csv(output_path, index=False, encoding="utf-8-sig")

print("结果行数：", len(subgroup_results))  # 应为 22
print("保存位置：", output_path.resolve())

customers["recency_group"] = pd.cut(
    customers["recency"],
    bins=[-float("inf"), 3, 6, 9, float("inf")],
    labels=["1-3个月内","4-6个月","7-9个月","10-12个月"],
) # 给recency分组，3个月一组
customers.to_csv(Path("Customers.csv"), index=False, encoding="utf-8-sig")

print("保存位置：", Path("Customers.csv").resolve())






