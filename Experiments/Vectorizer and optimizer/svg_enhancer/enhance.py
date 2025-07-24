import numpy as np
from svg.path import parse_path, Path, Line, CubicBezier, QuadraticBezier
from xml.dom import minidom

def calculate_angle(p1, p2, p3):
    # Convert tuples to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

def optimize_corners(points, angle_threshold=45):
    points = np.array(points)  # Convert input points to numpy array
    optimized = []
    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i-1], points[i], points[i+1])
        if angle < angle_threshold:
            # Preserve sharp corners
            optimized.append(points[i])
        else:
            # Smooth out gentle curves
            avg_point = (points[i-1] + points[i+1]) / 2
            optimized.append(avg_point)
    
    return np.array([points[0]] + optimized + [points[-1]])

def smooth_paths(points, smoothing_factor=0.2):
    points = np.array(points)  # Convert input points to numpy array
    smoothed = []
    for i in range(1, len(points) - 1):
        prev = points[i - 1]
        curr = points[i]
        next_pt = points[i + 1]
        
        # Calculate control points
        x = curr[0] + (next_pt[0] - prev[0]) * smoothing_factor
        y = curr[1] + (next_pt[1] - prev[1]) * smoothing_factor
        
        smoothed.append([x, y])
    
    return np.array([points[0]] + smoothed + [points[-1]])

def simplify_path(points, epsilon=1.0):
    points = np.array(points)  # Convert input points to numpy array
    
    def point_line_distance(point, start, end):
        if np.all(start == end):
            return np.linalg.norm(point - start)
        
        n = abs((end[1] - start[1]) * point[0] - 
                (end[0] - start[0]) * point[1] + 
                end[0] * start[1] - end[1] * start[0])
        d = np.sqrt((end[1] - start[1]) ** 2 + 
                    (end[0] - start[0]) ** 2)
        return n / d

    def recursive_simplify(points, epsilon, first, last):
        if last <= first + 1:
            return

        max_dist = 0
        max_idx = first
        
        for i in range(first + 1, last):
            dist = point_line_distance(points[i], 
                                     points[first], 
                                     points[last])
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > epsilon:
            recursive_simplify(points, epsilon, first, max_idx)
            recursive_simplify(points, epsilon, max_idx, last)
        else:
            for i in range(first + 1, last):
                points[i][2] = False  # Mark for removal

    mask = np.ones(len(points), dtype=bool)
    points_with_mask = np.column_stack((points, mask))
    
    recursive_simplify(points_with_mask, epsilon, 0, len(points_with_mask) - 1)
    return points_with_mask[points_with_mask[:, 2].astype(bool)][:, :2]

class SVGEnhancer:
    def __init__(self):
        self.params = {
            'simplify_epsilon': 1.0,
            'smooth_factor': 0.2,
            'angle_threshold': 45,
            'curve_error': 1.0
        }

    def extract_curves(self, path_element):
        curves = []
        d = path_element.getAttribute('d')
        path = parse_path(d)
        
        for segment in path:
            if isinstance(segment, CubicBezier):
                curves.append({
                    'type': 'cubic',
                    'start': (segment.start.real, segment.start.imag),
                    'control1': (segment.control1.real, segment.control1.imag),
                    'control2': (segment.control2.real, segment.control2.imag),
                    'end': (segment.end.real, segment.end.imag)
                })
            elif isinstance(segment, QuadraticBezier):
                curves.append({
                    'type': 'quadratic',
                    'start': (segment.start.real, segment.start.imag),
                    'control': (segment.control.real, segment.control.imag),
                    'end': (segment.end.real, segment.end.imag)
                })
            elif isinstance(segment, Line):
                curves.append({
                    'type': 'line',
                    'start': (segment.start.real, segment.start.imag),
                    'end': (segment.end.real, segment.end.imag)
                })
        return curves

    def enhance_curve(self, curve):
        if curve['type'] == 'cubic':
            points = np.array([
                curve['start'],
                curve['control1'],
                curve['control2'],
                curve['end']
            ])
            enhanced_points = self.enhance_bezier_points(points)
            return {
                'type': 'cubic',
                'start': tuple(enhanced_points[0]),
                'control1': tuple(enhanced_points[1]),
                'control2': tuple(enhanced_points[2]),
                'end': tuple(enhanced_points[3])
            }
        elif curve['type'] == 'quadratic':
            points = np.array([
                curve['start'],
                curve['control'],
                curve['end']
            ])
            enhanced_points = self.enhance_bezier_points(points)
            return {
                'type': 'quadratic',
                'start': tuple(enhanced_points[0]),
                'control': tuple(enhanced_points[1]),
                'end': tuple(enhanced_points[2])
            }
        else:  # line
            points = np.array([curve['start'], curve['end']])
            enhanced_points = self.enhance_line_points(points)
            return {
                'type': 'line',
                'start': tuple(enhanced_points[0]),
                'end': tuple(enhanced_points[1])
            }

    def enhance_bezier_points(self, points):
        enhanced = smooth_paths(points, self.params['smooth_factor'])
        enhanced = optimize_corners(enhanced, self.params['angle_threshold'])
        return enhanced

    def enhance_line_points(self, points):
        return simplify_path(points, self.params['simplify_epsilon'])

    def curve_to_path_data(self, curve):
        if curve['type'] == 'cubic':
            return f"C {curve['control1'][0]},{curve['control1'][1]} {curve['control2'][0]},{curve['control2'][1]} {curve['end'][0]},{curve['end'][1]}"
        elif curve['type'] == 'quadratic':
            return f"Q {curve['control'][0]},{curve['control'][1]} {curve['end'][0]},{curve['end'][1]}"
        else:  # line
            return f"L {curve['end'][0]},{curve['end'][1]}"

    def enhance_svg(self, input_file, output_file):
        doc = minidom.parse(input_file)
        paths = doc.getElementsByTagName('path')
        
        for path in paths:
            curves = self.extract_curves(path)
            enhanced_curves = [self.enhance_curve(curve) for curve in curves]
            
            new_path_data = f"M {enhanced_curves[0]['start'][0]},{enhanced_curves[0]['start'][1]}"
            for curve in enhanced_curves:
                new_path_data += " " + self.curve_to_path_data(curve)
            
            path.setAttribute('d', new_path_data)
        
        with open(output_file, 'w') as f:
            f.write(doc.toxml())
        doc.unlink()

# Usage
enhancer = SVGEnhancer()
enhancer.enhance_svg('../inputs/output3.svg', './output/enhanced.svg')