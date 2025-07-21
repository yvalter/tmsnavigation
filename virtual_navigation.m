function [arcLength1, arcLength2] = virtual_navigation(Head_circumference,Tragus_Tragus,Nasion_Inion, point)

%set up system of three equations (needed for ellipsoid formula)
P1=Head_circumference;
P2=Tragus_Tragus*1.4;
P3=Nasion_Inion*1.6;

syms a b c 
eqn1=P1== pi*(a+b)*(1+((3*(((a-b)^2)/((a+b)^2)))/(10+sqrt(4-3*(((a-b)^2)/((a+b)^2))))));
eqn2=P2== pi*(a+c)*(1+((3*(((a-c)^2)/((a+c)^2)))/(10+sqrt(4-3*(((a-c)^2)/((a+c)^2))))));
eqn3=P3== pi*(b+c)*(1+((3*(((b-c)^2)/((b+c)^2)))/(10+sqrt(4-3*(((b-c)^2)/((b+c)^2))))));

%solve ellipsoid formula
[a,b,c]=vpasolve([eqn1,eqn2,eqn3],[a,b,c]); 
sx=double(a/8.409);
sy=double(b/10.31);
sz=double(c/9.8157);
sx=sx*0.95; %adjustment factor to scalp width

fprintf('Loading MNI head...\n');
% Load unscaled MNI STL and apply transform
mni_unscaled = stlread('mni_unscaled.stl');
faces=mni_unscaled.faces;
[Vrt, ~, ic] = unique(mni_unscaled.vertices, 'rows');  % Remove duplicate vertices
F = ic(faces); % updates faces according to the new list of vertices

fprintf('individualizing head model...\n');
S=diag([sx, sy, sz, 1]); % creates a scaling matrix using the subject's approximated scalp radii
V_hom = [Vrt, ones(size(Vrt,1),1)];
V_transformed = (S * V_hom')'; % applies the scaling matrix to the vertices
V = V_transformed(:,1:3);  

fprintf('identifying scalp points...\n');
% Identifying target keypoints
unscaled_Cz=[0.33;-1.9828;94.6484;1];
unscaled_Fpz=[0.33;-103.347;-0.619;1];
unscaled_IP=[57.33;-76.05;-0.619;1];

% Applying scaling matrix to target keypoints
adjusted_point=[-point(1);-point(2);point(3);1];
scaled_point=(S*adjusted_point);
Target=[scaled_point(1:3)]';
scaled_Cz=(S*unscaled_Cz)';
Cz=scaled_Cz(1:3);
scaled_Fpz=(S*unscaled_Fpz);
Fpz=scaled_Fpz(1:3)';
scaled_IP=(S*unscaled_IP);
IP =scaled_IP(1:3)';

fprintf('calculating geodesic distances...\n');
% Define slicing planes
planeNormal1 = cross([0, 0, 0] - Target, Cz - Target);  % Plane 1: Cz-Target-0
planePoint1 = Target;

planeNormal2 = [0, 0, 1];                              % Plane 2: XY Plane
planePoint2 = Fpz;

% Intersect each plane with the mesh
[~, ~, arc1Full] = localArcLength(V, F, Cz, Target, planePoint1, planeNormal1);
[~, ~, arc2Full] = localArcLength(V, F, Fpz, Fpz, planePoint2, planeNormal2);


% Estimate intersection point (closest points between arcs)
D = pdist2(arc1Full, arc2Full);
[~, linearIdx] = min(D(:));
[idx1, idx2] = ind2sub(size(D), linearIdx);
IP = (arc1Full(idx1,:) + arc2Full(idx2,:)) / 2;

 % Arc 1: Cz to Target
 [arcLength1, seg1, ~] = localArcLength(V, F, Cz, Target, planePoint1, planeNormal1);

 % Arc 2: Fpz to IP
 [arcLength2, seg2, ~] = localArcLength(V, F, Fpz, IP, planePoint2, planeNormal2);

% Plotting
fprintf('generating figure...\n');
figure; hold on; axis equal; grid on;
title('TMS Virtual Navigation'); view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');

trisurf(F, V(:,1), V(:,2), V(:,3), ...
    'FaceAlpha', 0.5, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.8 0.85 1]);

view(45, 30);  % Set perspective separately

plot3(Cz(1), Cz(2), Cz(3), 'ro', 'LineWidth', 2);
plot3(Target(1), Target(2), Target(3), 'bo', 'LineWidth', 2);
plot3(Fpz(1), Fpz(2), Fpz(3), 'go', 'LineWidth', 2);
plot3(IP(1), IP(2), IP(3), 'mo', 'LineWidth', 2);

plot3(seg1(:,1), seg1(:,2), seg1(:,3), 'r-', 'LineWidth', 3);
plot3(seg2(:,1), seg2(:,2), seg2(:,3), 'm-', 'LineWidth', 3);

text(Cz(1), Cz(2), Cz(3)+8, 'Cz', 'FontWeight','bold');
text(Target(1)+5, Target(2), Target(3)+5, 'Target', 'FontWeight','bold');
text(Fpz(1)-5, Fpz(2)-15, Fpz(3)+5, 'Fpz', 'FontWeight','bold');
text(IP(1), IP(2), IP(3)+5, 'Intersection Point', 'FontWeight','bold');

legend({'Scalp Mesh','Cz','Target','Fpz','IP'}, 'Location','best');

fprintf('Arc distance from Fpz around head circumference: %.2f mm\n', arcLength2);
fprintf('Arc distance Cz → Target: %.2f mm\n', arcLength1);
end

%% 
function [arcLength, segmentPoints, orderedArc] = localArcLength(V, F, ptA, ptB, planePoint, planeNormal)
    intersectionLines = {};
    for i = 1:size(F,1)
        triVerts = V(F(i,:), :);
        [pts, isHit] = trianglePlaneIntersection(planePoint, planeNormal, triVerts);
        if isHit
            intersectionLines{end+1} = pts;
        end
    end

    if isempty(intersectionLines)
        arcLength = NaN; segmentPoints = []; orderedArc = [];
        return;
    end

    orderedArc = orderPolyline(vertcat(intersectionLines{:}));
    [projA, idxA] = projectToPolyline(ptA, orderedArc);
    [projB, idxB] = projectToPolyline(ptB, orderedArc);

    if idxA < idxB
        idxRange = idxA+1 : idxB-1;
        segmentPoints = [projA; orderedArc(idxRange,:); projB];
    else
        idxRange = idxB+1 : idxA-1;
        segmentPoints = [projB; orderedArc(idxRange,:); projA];
    end

    arcLength = sum(vecnorm(diff(segmentPoints), 2, 2));
end

%% 
function [intersectPts, isIntersecting] = trianglePlaneIntersection(planePoint, planeNormal, triVerts)
% TRIANGLEPLANEINTERSECTION computes the intersection between a triangle and a plane
% 
% Inputs:
%   planePoint  - A point on the plane [1x3]
%   planeNormal - Normal vector of the plane [1x3]
%   triVerts    - 3x3 matrix with triangle vertices as rows [v1; v2; v3]
%
% Outputs:
%   intersectPts    - Intersection points (0, 1, or 2 x 3 matrix)
%   isIntersecting  - True if the triangle intersects the plane

    % Tolerance for intersection
    tol = 1e-10;
    intersectPts = [];
    
    % Define triangle edges
    edges = [1 2; 2 3; 3 1];
    
    % Loop through edges
    for i = 1:3
        p0 = triVerts(edges(i,1), :);
        p1 = triVerts(edges(i,2), :);
        u = p1 - p0;
        w = p0 - planePoint;

        dotUN = dot(u, planeNormal);
        dotWN = dot(w, planeNormal);
        
        % Check if the edge is parallel to the plane
        if abs(dotUN) < tol
            continue
        end
        
        % Solve for intersection parameter
        s = -dotWN / dotUN;
        if s >= -tol && s <= 1+tol
            % Clamp s within [0, 1] to stay on the edge
            s = max(min(s,1),0);
            intersectPoint = p0 + s * u;
            intersectPts = [intersectPts; intersectPoint];
        end
    end
    
    % Remove duplicate points
    if size(intersectPts,1) > 1
        intersectPts = unique(round(intersectPts, 10), 'rows');
    end
    
    % Determine intersection validity
    isIntersecting = size(intersectPts,1) == 2;
end

%% 
function [projPt, projIdx] = projectToPolyline(pt, polyline)
    minDist = inf;
    projPt = pt;
    projIdx = 1;
    for i = 1:size(polyline,1)-1
        p0 = polyline(i,:);
        p1 = polyline(i+1,:);
        u = p1 - p0;
        s = dot(pt - p0, u) / dot(u,u);
        s = max(0, min(1, s));
        proj = p0 + s * u;
        d = norm(pt - proj);
        if d < minDist
            minDist = d;
            projPt = proj;
            projIdx = i;
        end
    end
end

%% 
function orderedPts = orderPolyline(pts)
    orderedPts = pts(1,:);
    pts(1,:) = [];
    while ~isempty(pts)
        dists = vecnorm(pts - orderedPts(end,:), 2, 2);
        [~, idx] = min(dists);
        orderedPts = [orderedPts; pts(idx,:)];
        pts(idx,:) = [];
    end
end