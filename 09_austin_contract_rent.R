# Use tidycensus to pull down contract rent data for the city of austin at the tract level 
# and visualize the results
library(tidycensus)
library(tidyverse)
library(sf)
# Set your Census API key (you can get one for free at https://api.census.gov/data/key_signup.html)
census_api_key("YOUR_CENSUS_API_KEY", install = TRUE)
# Define the variables we want to pull (these are from the ACS 5-year estimates)
variables <- c(
  total_units = "B25001_001",          # Total housing units
  occupied_units = "B25002_002",       # Occupied housing units
  vacant_units = "B25002_003",         # Vacant housing units
  median_rent = "B25064_001"           # Median gross rent
)
# Pull the data for Austin, TX at the tract level
austin_rent_data <- get_acs(
  geography = "tract",
  state = "TX",
  county = "Travis",
  variables = variables,
  geometry = TRUE
)

# Create a categorical variable for rent levels based on median rent using Jenks natural breaks, five categories
# Create the cut object separately so I can inspect the breakpoints and ensure they make sense for the Austin rent data distribution
rent_breaks <- classInt::classIntervals(austin_rent_data$estimate[austin_rent_data$variable == "median_rent"], n = 5, style = "jenks")
print(rent_breaks)


austin_rent_data <- austin_rent_data %>%
  group_by(variable) %>%
  mutate(rent_category = cut(estimate, breaks = rent_breaks$brks, 
  labels = c("Very Low ($<1518)", "Low ($1518-1916)", "Moderate ($1916-$2357)", "High ($2357-$2896)", "Very High (>$3501"), include.lowest = TRUE))

# Visualize the median rent across Austin tracts
ggplot(austin_rent_data) +
  geom_sf(aes(fill = rent_category), color = NA) +
  scale_fill_viridis_d(option = "plasma", na.value = "grey90") +
  ggthemes::theme_map() +
  labs(title = "Median Gross Rent by Census Tract in Austin, TX",
       fill = "Median Rent")
