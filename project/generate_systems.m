orders = [1, 2, 4, 8, 16, 32, 64, 128];
for i=1:1:length(orders)
    sys = rss(orders(i), 1, 1);
    while any(eig(sys.A) == 0)
        sys = rss(orders(i), 1, 1);
    end
    flag = 1;
    while flag
        try 
            are(sys.A, sys.B*sys.B', sys.C'*sys.C)
            flag = 0;
        catch
            sys = rss(orders(i), 1, 1);
        end
    end
    A = sys.A;
    B = sys.B;
    C = sys.C;
    save(sprintf("sys%d.mat", orders(i)), 'A', 'B', 'C');
    disp(orders(i))
end
