library(tidyverse)
library(stringr)
library(ggdendro)

colleges = read.delim("colleges.tsv", sep = "\t", header = TRUE)

# 1. In the clustering tutorial, we used k-means clustering to identify 
# 3 clusters of colleges using these criteria.

# 1.1. Replicate this analysis using the code in the tutorial to generate those 
# 3 clusters and append the cluster levels to the college_features dataset. 
# If you are getting an error using mutate, use the following code instead: 
# college_features$cluster = kmeans_cluster$cluster

# select relevant features
college_features = colleges %>% 
  select(institution_name, first_gen_share, poverty_rate, family_income_median, median_earnings, top_ten) %>% 
  na.omit() %>% distinct()

# run k-means clustering
kmeans_cluster = kmeans(select(college_features, -institution_name, -top_ten), 3)

# Find which cluster the observations belong to
#head(kmeans_cluster$cluster)

# append the cluster assignment onto the dataset
college_features = college_features %>% mutate(cluster = kmeans_cluster$cluster)
ggplot(college_features, aes(x = family_income_median, y = median_earnings, color = factor(cluster))) + 
  geom_point(alpha = 0.5) + 
  theme_minimal()

# 1.2. What is the median family income for each cluster 
# (hint: see kmeans_cluster$centers from the tutorial)?
kmeans_cluster$centers

# 1.3. Subset the colleges_features dataset on the cluster with the lowest 
# family_income_median, call this new data grant_candidates. 
# Note: in the tutorial, grant_candidates were from Cluster 1, 
# you could find that a different cluster from your analysis 
# has the lowest family_income_median when you look at kmeans_cluster$centers.

xx=which.min(kmeans_cluster$centers[,"family_income_median"])
xx
grant_candidates = college_features %>% filter(cluster == xx)

# 1.4. How many universities are in the cluster of grant receivers?
kmeans_cluster$size
kmeans_cluster$size[xx]

# 2. Upon review you’re informed that there are too many universities receiving grants. 
# The granting agency really likes the cluster approach but suggests you make 
# 5 clusters instead of 3.

# 2.1. Redo the k-means analysis above but create 5 clusters instead of 3. 
# Note: If you appended cluster onto your college_features dataset, 
# make sure to remove it before redoing the k-means analysis.

# select relevant features
college_features2 = colleges %>% 
  select(institution_name, first_gen_share, poverty_rate, family_income_median, median_earnings, top_ten) %>% 
  na.omit() %>% distinct()

# run k-means clustering
kmeans_cluster2 = kmeans(select(college_features2, -institution_name, -top_ten), 5)

# append the cluster assignment onto the dataset
college_features2 = college_features2 %>% mutate(cluster = kmeans_cluster2$cluster)

ggplot(college_features2, aes(x = family_income_median, y = median_earnings, color = factor(cluster))) + 
  geom_point(alpha = 0.5) + 
  theme_minimal()

# 2.2 Again subset the data on the cluster with the lowest family_income_median. 
# How many universities will receive a grant now? 
# What is the median and range of family_income_median of these 
# universities and how does it compare to your answers in Question 1?

kmeans_cluster2$centers
xx = which.min(kmeans_cluster2$centers[,"family_income_median"])
xx
grant_candidates2 = college_features2 %>% filter(cluster == xx)

kmeans_cluster2$size
kmeans_cluster2$size[xx]

median(grant_candidates2$family_income_median)
range(grant_candidates2$family_income_median)

median(grant_candidates$family_income_median)
range(grant_candidates$family_income_median)

# 2.3. You will likely find that there were two clusters out of the five
# with low but similar family_income_median. Among these two clusters, 
# what else determined which cluster these universities were assigned to 
# (hint: look at the centers again)? 
# Based on those other variables, do you think we made the correct decision 
# to distribute grants considering only family_income_median?

kmeans_cluster2$centers

# 3. Hierarchical clustering: Part of the grant is to reformulate curriculums
# to better match top ten universities.

# 3.1. Subset your colleges dataset using the following code. 
# The !is.na(sat_verbal_quartile_1) removes universities that do not have SAT admission criteria,
# so we are looking at similar degree-granting universities. 
# What other criteria are we using to subset? 

grant_colleges = colleges %>% 
  filter((!is.na(sat_verbal_quartile_1) & family_income_median < 40000 & median_earnings < 30000)) 

top_ten_schools = colleges %>% 
  filter(top_ten == TRUE)

heir_analysis_data = rbind(grant_colleges, top_ten_schools)

# 3.2. Replicate the heirarchical clustering from the tutorial comparing 
# major percentages using heir_analysis_data dataset. 
# Which universities are the most different from the top ten schools 
# in terms of majors?

# select all the columns that contain the string '_major_perc'
major_perc = heir_analysis_data %>% 
  select(institution_name, top_ten, str_which(names(heir_analysis_data), "_major_perc")) %>% 
  na.omit()

# compute the euclidean distance
euclidean = dist(select(major_perc, -institution_name, -top_ten), method = "euclidean")
# hierarchical clustering
hier = hclust(euclidean) # labels are ids
hier$labels

# replace labels with institution name
hier$labels = major_perc$institution_name
# plot dendrogram
ggdendrogram(hier, rotate = TRUE, size = 2)

# color the leafs on the dendro by top_ten
# extract the dendrogram data
dendro_data = dendro_data(hier)

dendro_data$labels = unique(merge(dendro_data$labels,
                                  select(college_features, institution_name,
                                         top_ten), by.x = "label", by.y = "institution_name",
                                  all.x = TRUE))

ggplot(segment(dendro_data)) + 
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
  geom_text(data = label(dendro_data), aes(label = label, x = x, y = 0, hjust = 0, color = top_ten), size = 2) + 
  coord_flip() + 
  scale_y_reverse(expand = c(0.25, 0)) + 
  theme_minimal() + 
  theme(legend.position = "bottom")

# schools that are the most different with top ten schools include:
hier$labels[hier$order[c(1:9)]]

# 3.3. How else can we compare the grantee schools to the top ten schools? 
# Explore using any of the methods we learned in this class.

# Additional ways to compare grantee schools to the top ten schools:
# 1. Major percentages could be examined using k-means clustering
# 2. Other curriculum-related features could be examined using hierarchical clustering

# 1. run k-means clustering on the major percentages
# 5 clusters
kmeans_cluster10 = kmeans(select(major_perc, -institution_name, -top_ten), 5)
# append the cluster assignment onto the dataset
major_perc10 = major_perc %>% mutate(cluster = kmeans_cluster10$cluster)

# ggplot(major_perc10, aes(x = liberal_arts_major_perc, y = engineering_major_perc, color = factor(cluster))) + 
#   geom_point(alpha = 0.5) + 
#   theme_minimal()

major_perc10[which(major_perc10$top_ten==TRUE),"cluster"]
kmeans_cluster10$size
major_perc10[which(major_perc10$cluster==1),"institution_name"]
major_perc10[which(major_perc10$top_ten==FALSE & major_perc10$cluster==1),"institution_name"]
major_perc10[which(major_perc10$top_ten==TRUE & major_perc10$cluster!=1),"institution_name"]

# 2. run hierarchical clustering based on other curriculum-related variables 
# (all selected variables are comparable in fraction between 0-1)
heir_analysis_data2 = heir_analysis_data %>% 
  select(institution_name, top_ten, part_time_percent,
         pell_grant_rate, retention_rate, federal_loan_rate,
         loan_ever, pell_ever) %>% 
  na.omit()

euclidean2 = dist(select(heir_analysis_data2, -institution_name, -top_ten), method = "euclidean")
# hierarchical clustering
hier2 = hclust(euclidean2) # labels are ids

# replace labels with institution name
hier2$labels = heir_analysis_data2$institution_name
# plot dendrogram
ggdendrogram(hier2, rotate = TRUE, size = 2)

# color the leafs on the dendro by top_ten
# extract the dendrogram data
dendro_data2 = dendro_data(hier2)

dendro_data2$labels = unique(merge(dendro_data2$labels,
                                  select(college_features, institution_name,
                                         top_ten), by.x = "label", by.y = "institution_name",
                                  all.x = TRUE))

ggplot(segment(dendro_data2)) + 
  geom_segment(aes(x = x, y = y, xend = xend, yend = yend)) + 
  geom_text(data = label(dendro_data2), aes(label = label, x = x, y = 0, hjust = 0, color = top_ten), size = 2) + 
  coord_flip() + 
  scale_y_reverse(expand = c(0.25, 0)) + 
  theme_minimal() + 
  theme(legend.position = "bottom")
