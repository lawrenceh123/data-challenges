library(tidyverse)
library(GGally)

ff = read.delim('forestfires.tsv', header = TRUE, sep = '\t')
ff$log_area = log10(ff$area+1)

ggpairs(select(ff, temp, wind, area), title = 'Relationship between temp, wind, and area')

ggpairs(select(ff, RH, rain, ISI, area), title = 'Relationship between RH, rain, ISI, and area')

# temp and area
ggplot(ff, aes(x = temp, y = area)) +
  geom_point() + 
  geom_smooth(method = "lm") +
  scale_y_log10() +
  ggtitle("area vs temp")

# wind and area
ggplot(ff, aes(x = wind, y = area)) + 
  geom_point() + 
  geom_smooth(method = "lm") +
  scale_y_log10() +
  ggtitle("area vs wind")

# RH and area
ggplot(ff, aes(x = RH, y = area)) + 
  geom_point() + 
  geom_smooth(method = "lm") +
  scale_y_log10() +
  ggtitle("area vs RH")

# rain and area
ggplot(ff, aes(x = rain, y = area)) + 
  geom_point() + 
  geom_smooth(method = "lm") +
  scale_y_log10() +
  ggtitle("area vs rain")

# ISI and area
ggplot(ff, aes(x = ISI, y = area)) + 
  geom_point() + 
  geom_smooth(method = "lm") +
  scale_y_log10() +
  ggtitle("area vs ISI")


#order of factor
ff$month = factor(ff$month, levels = c('jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
#mean burn area by month
area_by_month = ff %>%
  group_by(month) %>%
  summarize(n = n(), mean_area = mean(area), sd_area = sd(area))

limits = aes(ymax = area_by_month$mean_area + area_by_month$sd_area, ymin = area_by_month$mean_area - area_by_month$sd_area)
area_by_month %>% ggplot(mapping = aes(x = month, y = mean_area)) + 
  geom_bar(stat = 'identity') + 
  geom_errorbar(limits) + 
  ggtitle("area by month (mean +/- sd)")


ggpairs(select(ff, temp, RH, wind, log_area), title = 'Relationship between temp, RH, wind, and log_area')

ggpairs(select(ff, temp, RH, wind, log_area) %>%
          filter(log_area > 0), title = 'Relationship between temp, RH, wind, and log_area > 0')
