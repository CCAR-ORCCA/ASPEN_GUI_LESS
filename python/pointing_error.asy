import geometry;
size(8cm);
settings.outformat = "pdf";

real f = 0.1;
real r = 0.4;
real Delta = 0.2;
real h = 0.3;

pair A = (0, 0.2);
pair O = (0, 0);
pair C = (0, -0.2);
pair D = (f, 0);
pair E = (r, 0);
pair F = (r, -Delta);
pair H = (0, h);



pair[] intersect = intersectionpoints(A -- C, F -- (F + 4 * (D - F)));
pair B = intersect[0];


draw(A -- C);
draw(O -- E);
draw(F -- E);
draw(F -- B, dashed);
draw(O -- H, 1bp+dashed,Arrow);


distance("$\delta$", O , B, -5mm, rotated = false);
distance("$r$", E , D, -5mm);
distance("$f$", O, D, 5mm);
distance("$\Delta$", E , F, -5mm, rotated = false);



dot(D);











