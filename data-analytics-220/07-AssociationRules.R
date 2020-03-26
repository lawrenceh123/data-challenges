library(tidyverse)
library(GGally)
library(arules)
library(arulesViz)

# 1. Read in the data using the read.delim function. 
# Then use ggplot2, ggpairs, and dplyr to identify interesting relationships in the data. 
# Write a short description of one interesting pattern you identified.

colleges = read.delim("colleges.tsv", sep = "\t", header = TRUE)

# exploratory data analysis using ggplot2, ggpairs, and dplyr
ggplot(colleges, aes(x = cost, fill = control)) +
  #geom_histogram(aes(y = ..density..)) 
  geom_histogram(alpha = 0.6) +
  facet_grid(control~.) +
  ggtitle("histogram of cost by control")

ggplot(colleges, aes(x = control, y = cost)) +
  geom_boxplot() +
  ggtitle("cost by control")

# Observation: 
# Median cost for both private for-profit and nonprofit colleges are higher than public colleges.
# The median cost of private for-profit colleges is slightly lower but comparable to private non-profit colleges.  
# The cost distributions between the two look comparable.

ggplot(colleges, aes(x = control, y = retention_rate)) +
  geom_boxplot() +
  ggtitle("retention rate by control")

ggplot(colleges, aes(x = control, y = age_entry_avg)) +
  geom_boxplot() +
  ggtitle("avg age entry by control")

ggplot(colleges, aes(x = control, y = female_share)) +
  geom_boxplot() +
  ggtitle("female share by control")

ggplot(colleges, aes(x = control, y = poverty_rate)) +
  geom_boxplot() +
  ggtitle("poverty rate by control")

# Additional comparisons based on control:
# female_share (and to a lesser extent, age_entry_avg, married_share) 
# may be higher in private for-profit colleges, whereas retention_rate may be lower.

ggpairs(select(colleges, control, cost, median_debt, median_earnings, first_gen_share, family_income_median),
        mapping = aes(color = control, alpha = 0.5),
        title = 'Relationship between several cost/income-associated variables')

# One interesting pattern: College cost is positively correlated with median debt, median earnings, median family income, 
# and is negatively correlated with first generation share. 
# These correlations are the strongest for public colleges. 
# This suggests that colleges (and in particular public colleges) with higher cost generally have 
# higher median debt, higher median earnings, lower first gen share, and higher median family income. 
# Median cost for public colleges is lower than private for-profit or private non-profit colleges.

# 2. Prepare your data for association rule mining by transforming it into a set of transactions. 
# Use the inspect and summary functions to view the transactions.

# discretize numeric features of interest before transforming into transactions
colleges$cost_quartiles = discretize(colleges$cost,
                                     method = "frequency",
                                     categories = 4,
                                     labels = c("cost_Q1", "cost_Q2", "cost_Q3", "cost_Q4"))

colleges$earnings_quartiles = discretize(colleges$median_earnings,
                                         method = "frequency", 
                                         categories = 4, 
                                         labels = c("earnings_Q1", "earnings_Q2", "earnings_Q3", "earnings_Q4"))

colleges$debt_quartiles = discretize(colleges$median_debt,
                                     method = "frequency", 
                                     categories = 4, 
                                     labels = c("debt_Q1", "debt_Q2", "debt_Q3", "debt_Q4"))

colleges$age_quartiles  = discretize(colleges$age_entry_avg,
                                     method = "frequency", 
                                     categories = 4, 
                                     labels = c("age_Q1", "age_Q2", "age_Q3", "age_Q4"))

colleges$female_quartiles  = discretize(colleges$female_share,
                                        method = "frequency", 
                                        categories = 4, 
                                        labels = c("female_Q1", "female_Q2", "female_Q3", "female_Q4"))

colleges$married_quartiles  = discretize(colleges$married_share,
                                         method = "frequency", 
                                         categories = 4, 
                                         labels = c("married_Q1", "married_Q2", "married_Q3", "married_Q4"))

colleges$veteran_quartiles  = discretize(colleges$veteran_share,
                                         method = "frequency", 
                                         categories = 4, 
                                         labels = c("veteran_Q1", "veteran_Q2", "veteran_Q3", "veteran_Q4"))

colleges$first_gen_quartiles  = discretize(colleges$first_gen_share,
                                           method = "frequency", 
                                           categories = 4, 
                                           labels = c("first_gen_Q1", "first_gen_Q2", "first_gen_Q3", "first_gen_Q4"))

colleges$fam_income_quartiles  = discretize(colleges$family_income_median,
                                            method = "frequency", 
                                            categories = 4, 
                                            labels = c("fam_income_Q1", "fam_income_Q2", "fam_income_Q3", "fam_income_Q4"))

colleges$born_us_quartiles  = discretize(colleges$pct_born_us,
                                         method = "frequency", 
                                         categories = 4, 
                                         labels = c("born_us_Q1", "born_us_Q2", "born_us_Q3", "born_us_Q4"))

colleges$poverty_quartiles  = discretize(colleges$poverty_rate,
                                         method = "frequency", 
                                         categories = 4, 
                                         labels = c("poverty_Q1", "poverty_Q2", "poverty_Q3", "poverty_Q4"))

colleges$unemployment_quartiles  = discretize(colleges$unemployment_rate,
                                              method = "frequency", 
                                              categories = 4, 
                                              labels = c("unemployment_Q1", "unemployment_Q2", "unemployment_Q3", "unemployment_Q4"))

colleges$retention_quartiles  = discretize(colleges$retention_rate,
                                           method = "frequency", 
                                           categories = 4, 
                                           labels = c("retention_Q1", "retention_Q2", "retention_Q3", "retention_Q4"))

# estimate percentage of STEM majors and indicatie whether a school has a large proportion of STEM students
colleges = colleges %>% mutate(stem_perc = architecture_major_perc + comm_tech_major_perc +
                                 computer_science_major_perc + engineering_major_perc +
                                 eng_tech_major_perc + bio_science_major_perc + math_stats_major_perc,
                               high_stem = ifelse(stem_perc >= 0.3, TRUE, FALSE))

# select interesting categorical data to mine
college_features = colleges %>% select(locale, control, pred_deg, historically_black, men_only,
                                       women_only, religious, online_only, top_ten, cost_quartiles, earnings_quartiles,
                                       debt_quartiles, age_quartiles, female_quartiles, married_quartiles, veteran_quartiles,
                                       first_gen_quartiles, fam_income_quartiles, born_us_quartiles, poverty_quartiles, 
                                       unemployment_quartiles, retention_quartiles, high_stem)

# load data into a transaction object
college_trans = as(college_features, "transactions")
# inspect and summary to view the transactions.
inspect(college_trans[1:3])
summary(college_trans)

# plot the most frequent items
#itemFrequencyPlot(college_trans, topN = 10, cex = 0.7)

# 3. Generate rules with the apriori function with a support of 0.01 and a confidence of 0.60. 
rules = apriori(college_trans, parameter = list(sup = 0.01, conf = 0.6, target = "rules"))

summary(rules)
items(rules)

# view the rules
inspect(head(rules))

# sort the rules by lift
inspect(head(sort(rules, by = "lift")))

# view quality metrics
head(quality(rules))

# 4. Try the following combinations of support and confidence: [0.10, 0.60], [0.01, 0.10]. 
# What happens to the number of rules as the support increases? 
# (Hint: use the summary function to see the number of rules.)

rules2 = apriori(college_trans, parameter = list(sup = 0.1, conf = 0.6, target = "rules"))
summary(rules2)
rules3 = apriori(college_trans, parameter = list(sup = 0.01, conf = 0.1, target = "rules"))
summary(rules3)
# The number of rules decreases as minimum support requirement increases (from 0.01 to 0.1). 
# (Whereas the number of rules increases as minimum confidence requirement decreases (from 0.6 to 0.1).)

# 5. In the text we constructed earnings quartiles and explored the associations in top earners 
# by filtering the rules for the top quartile of earners. 
# Now, re-filter the rules to explore the bottom 25% of earners (Q1). 
# Report at least 1 interesting finding. 
# Hint: Use the subset and inspect functions to filter the right-hand side (rhs) for 'earnings_quartiles=earnings_Q1'.

low_earners = subset(rules, subset = rhs %in% "earnings_quartiles=earnings_Q1" & lift > 1)

# top 10 rules by lift
inspect(head(low_earners, n = 10, by = "lift"))
#
# plot top 10 rules by lift
plot(sort(low_earners, by = "lift")[1:10], method = "graph")
