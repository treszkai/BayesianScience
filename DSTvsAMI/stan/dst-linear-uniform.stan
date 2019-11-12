data {
  int DAYS;            // Number of days
  int STUDIES;         // Number of studies

  // The observed AMI counts and the trend predictions, for each day of each study
    int<lower=0> ami_obs[STUDIES, DAYS];
  real<lower=0> ami_trend[STUDIES, DAYS];
}

parameters {
  real rr_Mon_minus_1;
}

transformed parameters {
  real rr_day[DAYS];
  real ami_dst_mean[STUDIES, DAYS];

  for (i in 1:DAYS) {
    rr_day[i] = (rr_Mon_minus_1 * (DAYS + 1 - i) / DAYS) + 1;
  }

  for (s in 1:STUDIES) {
    for (i in 1:DAYS) {
      ami_dst_mean[s][i] = ami_trend[s][i] * rr_day[i];
    }
  }
}

model {
  for (s in 1:STUDIES) {
    for (i in 1:DAYS) {
      ami_obs[s][i] ~ poisson(ami_dst_mean[s][i]);
    }
  }
}
