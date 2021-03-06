data {
  int DAYS;            // Number of days
  int STUDIES;         // Number of studies
  real NORMAL_SIGMA;   // The standard deviation of the normal component of the prior
  real EXPON_BETA;     // The beta parameter of the exponential component of the prior

  // The observed AMI counts and the trend predictions, for each day of each study
  int<lower=0> ami_obs[STUDIES, DAYS];
  real<lower=0> ami_trend[STUDIES, DAYS];
}

parameters {
  // Monday RR - 1.
  // (We cannot model RR_Mon directly because cannot assign a
  //   common distribution for that.)
  // Its probabilistic value is assigned in the model block below.
  real rr_Mon_minus_1;
}

transformed parameters {
  // The RR for every day
  real rr_day[DAYS];
  // The posttransitional AMI counts for every day of every study.
  real ami_dst_mean[STUDIES, DAYS];

  // Specifying the RR for every day, using the linear weekday model.
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
  // Mixture models are specified using the construct below:
  // target += log_sum_exp(c1 * XXX_lpdf(x | p1), c2 * YYY_lpdf(x | p2));
  target += log_sum_exp(normal_lpdf(rr_Mon_minus_1 | 0, NORMAL_SIGMA),
                        exponential_lpdf(rr_Mon_minus_1 | EXPON_BETA));

  // Finally, the observations are drawn from a Poisson distribution.
  for (s in 1:STUDIES) {
    for (i in 1:DAYS) {
      ami_obs[s][i] ~ poisson(ami_dst_mean[s][i]);
    }
  }
}
