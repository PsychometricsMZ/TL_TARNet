library(magrittr)
library(dplyr)
library(stringr)

load(".../ICPSR/DS0002/36151-0002-Data.rda")
data2 <- da36151.0002
load(".../ICPSR/DS0001/36151-0001-Data.rda")
data1 <- da36151.0001
load(".../ICPSR/DS0012/36151-0012-Data.rda")
data12 <- da36151.0012


merged01 <- data12 %>%
  left_join(data2, by = c("DISTID","PSUID","STATEID"))

data <- data1%>%
  left_join(merged01, by = "IDHH")

data_clean <- data
data_clean  <- data %>% 
  filter(between(RO5, 6, 18),          # child age 
         CS3Y == "(1) Yes 1",          # currently enrolled in school 
         URBAN2011.x == "(0) rural 0") 



data_clean <- data_clean %>% 
  mutate(
    fu7  = as.numeric(str_extract(as.character(FU7),  "\\d+$")),
    fu7a = as.numeric(str_extract(as.character(FU7A), "\\d+$")),
    
    collect_fw = case_when(
      fu7  %in% c(2, 5) & fu7a %in% c(2, 3, 4) ~ 1,   # cooks with firewood AND collects it
      fu7  %in% c(2, 5) & fu7a == 1             ~ 0,   # cooks with firewood but purchases it
      TRUE                                      ~ 0    # all other households
    )
  )


## 5.1  Study minutes per week
data_clean <- data_clean %>% 
  mutate(study_min_wk = 60 * (CS10 + CS11))   # hours/week to minutes


data_clean <- data_clean %>% 
  mutate(
    TA9B_num = as.numeric(str_extract(as.character(TA9B), "\\d+$")),      
    
    ## maths-proficiency dummy: 1 if code ≥ 2 (“subtract” or “divide”),
    ##                           0 if code < 2,  NA if missing / DK
    math_ok  = case_when(
      is.na(TA9B_num)        ~ NA_real_,  
      TA9B_num >= 2          ~ 1,
      TRUE                   ~ 0
    )
  )


data_clean <- data_clean %>%
  mutate(
    ln_income = case_when(
      INCOME.x >= 0 ~ log1p(INCOME.x),
      TRUE          ~ NA_real_
    )
  )

data_clean <- data_clean %>%
  mutate(
    mom_edu_code = as.numeric(str_extract(as.character(HHEDUCF.x), "\\d+$")),
    mother_edu = case_when(
      is.na(mom_edu_code) | mom_edu_code == 0        ~ 1,  # Illiterate (reference)
      mom_edu_code >= 1  & mom_edu_code <= 4         ~ 2,  # 1–4 std.
      mom_edu_code >= 5  & mom_edu_code <= 9         ~ 3,  # 5–9 std.
      mom_edu_code >= 10 & mom_edu_code <= 11        ~ 4,  # 10–11 std.
      mom_edu_code >= 12                             ~ 5   # 12th & college or above
    )
  )


data_clean <- data_clean %>%
  mutate(
    caste_code    = as.numeric(str_extract(as.character(ID13.x), "\\d+$")),
    religion_code = as.numeric(str_extract(as.character(ID11.x), "\\d+$")),
    # Collapse into 6-category caste-religion group
    caste_religion_group = case_when(
      # 1 = Forward caste Hindus (Brahmin or General caste AND Hindu)
      religion_code == 1 & caste_code %in% c(1, 2)       ~ 1,
      # 2 = OBC Hindus
      religion_code == 1 & caste_code == 3               ~ 2,
      # 3 = SC Hindus
      religion_code == 1 & caste_code == 4               ~ 3,
      # 4 = ST Hindus
      religion_code == 1 & caste_code == 5               ~ 4,
      # 5 = Muslim (regardless of caste)
      religion_code == 2                                 ~ 5,
      # 6 = Christian, Sikh, Jain, Buddhist, others
      religion_code %in% c(3, 4, 5, 6)                    ~ 6,
      # Otherwise unknown
      TRUE                                               ~ NA_real_
    )
  )

data_clean$elec <- ifelse(as.numeric(data_clean$FU1) == 1, 1L, 0L)

data_clean$poor <- ifelse(
  data_clean$POOR == "(1) poor 1",
  1L, 
  0L
)

names(data_clean)[names(data_clean) == "CS5" ] <- "dist_school_km"


names(data_clean)[names(data_clean) == "STATEID.x" ] <- "state"

vars1 <- c("ln_income", "mother_edu","caste_religion_group","elec","dist_school_km","poor", "state","collect_fw", "study_min_wk","math_ok")
data_clean2 <- data_clean[vars1]

ihds_data <- na.omit(data_clean2)


