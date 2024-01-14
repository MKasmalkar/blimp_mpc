clc
close all

cs_lp_20_err = error_norm(cs_lp_n20);
cs_lp_40_err = error_norm(cs_lp_n40);
cs_lp_60_err = error_norm(cs_lp_n60);
cs_lp_80_err = error_norm(cs_lp_n80);
cs_lp_100_err = error_norm(cs_lp_n100);

cs_lp_errs = [cs_lp_20_err
              cs_lp_40_err
              cs_lp_60_err
              cs_lp_80_err
              cs_lp_100_err];

cs_nlp_20_err = error_norm(cs_nlp_n20);
cs_nlp_40_err = error_norm(cs_nlp_n40);
cs_nlp_60_err = error_norm(cs_nlp_n60);
cs_nlp_80_err = error_norm(cs_nlp_n80);
cs_nlp_100_err = error_norm(cs_nlp_n100);

cs_nlp_errs = [cs_nlp_20_err
               cs_nlp_40_err
               cs_nlp_60_err
               cs_nlp_80_err
               cs_nlp_100_err];

cs_lp_20_st = mean(cs_lp_n20(:, 34));
cs_lp_40_st = mean(cs_lp_n40(:, 34));
cs_lp_60_st = mean(cs_lp_n60(:, 34));
cs_lp_80_st = mean(cs_lp_n80(:, 34));
cs_lp_100_st = mean(cs_lp_n100(:, 34));

cs_lp_sts = [cs_lp_20_st
             cs_lp_40_st
             cs_lp_60_st
             cs_lp_80_st
             cs_lp_100_st];

cs_nlp_20_st = mean(cs_nlp_n20(:, 34));
cs_nlp_40_st = mean(cs_nlp_n40(:, 34));
cs_nlp_60_st = mean(cs_nlp_n60(:, 34));
cs_nlp_80_st = mean(cs_nlp_n80(:, 34));
cs_nlp_100_st = mean(cs_nlp_n100(:, 34));

cs_nlp_sts = [cs_nlp_20_st
              cs_nlp_40_st
              cs_nlp_60_st
              cs_nlp_80_st
              cs_nlp_100_st];

cs_lp_20_phi = rms(cs_lp_n20(:, 5));
cs_lp_40_phi = rms(cs_lp_n40(:, 5));
cs_lp_60_phi = rms(cs_lp_n60(:, 5));
cs_lp_80_phi = rms(cs_lp_n80(:, 5));
cs_lp_100_phi = rms(cs_lp_n100(:, 5));

cs_lp_phi = [cs_lp_20_phi
             cs_lp_40_phi
             cs_lp_60_phi
             cs_lp_80_phi
             cs_lp_100_phi] * 180/pi;

cs_nlp_20_phi = rms(cs_nlp_n20(:, 5));
cs_nlp_40_phi = rms(cs_nlp_n40(:, 5));
cs_nlp_60_phi = rms(cs_nlp_n60(:, 5));
cs_nlp_80_phi = rms(cs_nlp_n80(:, 5));
cs_nlp_100_phi = rms(cs_nlp_n100(:, 5));

cs_nlp_phi = [cs_nlp_20_phi
              cs_nlp_40_phi
              cs_nlp_60_phi
              cs_nlp_80_phi
              cs_nlp_100_phi] * 180/pi;

cs_lp_20_th = rms(cs_lp_n20(:, 6));
cs_lp_40_th = rms(cs_lp_n40(:, 6));
cs_lp_60_th = rms(cs_lp_n60(:, 6));
cs_lp_80_th = rms(cs_lp_n80(:, 6));
cs_lp_100_th = rms(cs_lp_n100(:, 6));

cs_lp_th  = [cs_lp_20_th
             cs_lp_40_th
             cs_lp_60_th
             cs_lp_80_th
             cs_lp_100_th] * 180/pi;

cs_nlp_20_th = rms(cs_nlp_n20(:, 6));
cs_nlp_40_th = rms(cs_nlp_n40(:, 6));
cs_nlp_60_th = rms(cs_nlp_n60(:, 6));
cs_nlp_80_th = rms(cs_nlp_n80(:, 6));
cs_nlp_100_th = rms(cs_nlp_n100(:, 6));

cs_nlp_th = [cs_nlp_20_th
             cs_nlp_40_th
             cs_nlp_60_th
             cs_nlp_80_th
             cs_nlp_100_th] * 180/pi;

n = [20, 40, 60, 80, 100];

subplot(411)
bar(n, [cs_lp_errs, cs_nlp_errs]);
title("Mean position error")
xlabel('Time horizon (time steps)')
ylabel('Mean error (m)')
legend('Linear MPC', 'Nonlinear MPC')

subplot(412)
bar(n, [cs_lp_sts ./ 1e9, cs_nlp_sts ./ 1e9]);
title("Mean solve time")
xlabel('Time horizon (time steps)')
ylabel('Solve time (sec)')

subplot(413)
bar(n, [cs_lp_phi, cs_nlp_phi])
title("RMS phi error")
xlabel('Time horizon (time steps)')
ylabel('Phi (deg)')

subplot(414)
bar(n, [cs_lp_th, cs_nlp_th])
title("RMS theta error")
xlabel('Time horizon (time steps)')
ylabel('Theta (deg)')

function err = error_norm(mat)
    abs_error = sqrt(mat(:, 30).^2 + mat(:, 31).^2 + mat(:, 32).^2);
    err = mean(abs_error);
end