T1 = readtable('t1_Square_ppo_.csv');
T2 = readtable('t1_NoSquare_ppo_.csv');
Sx1 = T1.x1;
Sx2 = T1.x2;
Sx3 = T1.x3;
Sx4 = T1.x4;
NSx1 = T2.x1;
NSx2 = T2.x2;
NSx3 = T2.x3;
NSx4 = T2.x4;

figure(1)
subplot(2,2,1)
plot(Sx1,'Linewidth',2);
hold on
plot(NSx1,'Linewidth',2);
title('x1')
legend('Square','NoSquare')
grid on
subplot(2,2,2)
plot(Sx2,'Linewidth',2);
hold on
plot(NSx2,'Linewidth',2);
title('x2')
legend('Square','NoSquare')
grid on
subplot(2,2,3)
plot(Sx3,'Linewidth',2);
hold on
plot(NSx3,'Linewidth',2);
title('x3')
legend('Square','NoSquare')
grid on
subplot(2,2,4)
plot(Sx4,'Linewidth',2);
hold on
plot(NSx4,'Linewidth',2);
title('x4')
legend('Square','NoSquare')
grid on