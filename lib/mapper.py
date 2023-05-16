"""Plots 3D environment.

AA229: Sequential Decision Making final project code.
"""

__authors__ = "D. Knowles"
__date__ = "27 Oct 2020"

import numpy as np
import pyqtgraph as pg
import pyqtgraph.Vector as Vector
import pyqtgraph.opengl as gl
import scipy.linalg as la
from PyQt5 import QtWidgets

import lib.parameters as P
from lib.tools import RotationBody2Vehicle


# temp


class WorldMapper():
    def __init__(self):
        self.num_agents = P.agent_states.shape[1]

        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Multi-Robot POMDP')
        self.window.setGeometry(0, 0, 500, 500)  # args: upper_left_x, upper_right_y, width, height
        sg = QtWidgets.QDesktopWidget().availableGeometry()
        # self.window.setGeometry(sg.width()/2.,0,sg.width()/2.,sg.height())
        self.window.opts['center'] = Vector((P.grid_x[1] + P.grid_x[0]) / 2.,
                                            (P.grid_y[1] + P.grid_y[0]) / 2.,
                                            0.)
        # self.window.opts['center'] = Vector(0.,0.,0.)
        # self.window.setCameraPosition(distance=P.grid_x[1] - P.grid_x[0], elevation=60, azimuth=-90)
        self.window.setCameraPosition(distance=100., elevation=90, azimuth=-90)

        self.window.show()  # display configured window
        self.window.raise_()  # bring window to the front
        self.agents_initialized = False  # have the agents been plotted yet?
        self.plans_initialized = False  # have the plans been plotted yet?
        self.paths_initialized = False  # have the paths been plotted yet?
        self.state_history = []
        self.estimate_history = []
        # get points that define the non-rotated, non-translated mav and the mesh colors
        vertex_names = ["small_rover_body_vertices.csv",
                        "small_rover_wheels_vertices.csv"]
        self.agent_points = self.get_agent_points(vertex_names)
        large_vertices = ["large_rover_body_vertices.csv",
                          "large_rover_wheels_vertices.csv",
                          "large_rover_struts_vertices.csv",
                          ]
        self.large_points = self.get_agent_points(large_vertices)

        # dubins path parameters
        self.agents = []
        self.agent_colors = [np.array([1., 0., 0., 1]),
                             np.array([1.0, 0.647, 0., 1]),
                             np.array([0., 1., 0., 1]),
                             np.array([0., 0., 1., 1]),
                             np.array([0.578, 0., 0.827, 1])]
        self.draw_map()

        self.uc = 0  # update counter for viewer functions

        self.app.processEvents()

    ###################################
    # public functions
    def update(self, agent_states, estimated_states, covs=[None], plans=[None]):
        """ Update visualization

        Parameters
        ----------
        agent_states : np.ndarray
            States of all agents where of size (#states x #agents).
        estimated_states : np.ndarray
            Estimated states of all agents where of size
            (#states x #agents).
        covs : list of tuples
            Covariances for each agent
        plans :
            Future plans that are optionally also drawn.

        """

        history_timestep = np.expand_dims(agent_states.copy(), 2)
        estimate_timestep = np.expand_dims(estimated_states.copy(), 2)
        if len(self.state_history) == 0:
            self.state_history = history_timestep
            self.estimate_history = estimate_timestep
        else:
            self.state_history = np.concatenate((self.state_history,
                                                 history_timestep), axis=2)
            self.estimate_history = np.concatenate((self.estimate_history,
                                                    estimate_timestep), axis=2)

        # initialize the drawing the first time update() is called
        self.draw_agents(agent_states)
        if not self.agents_initialized:
            self.agents_initialized = True

        # update plans if they exist
        if len(plans) == 1 and plans[0] == None:
            pass
        else:
            self.draw_plan(plans)

        # update plans if they exist
        if len(covs) == 1 and np.any(covs[0]) == None:
            pass
        else:
            self.draw_ellipses(covs)

        # update state drawing
        self.draw_paths()

        # update the center of the camera view to the mav location
        # view_location = Vector(agent_states[0,P.sim_agent_idx],
        #                        agent_states[1,P.sim_agent_idx],
        #                        agent_states[2,P.sim_agent_idx])  # defined in ENU coordinates
        # circle middle
        # er = 1000.
        # sf = 0.01
        # view_location = Vector(P.bldg_e_concentration + er*np.cos(sf*self.uc),
        #                        P.bldg_n_concentration + er*np.sin(sf*self.uc),
        #                        500.)  # defined in ENU coordinates
        # view_location = Vector(P.bldg_e_concentration,
        #                        P.bldg_n_concentration,
        #                        200.)  # defined in ENU coordinates
        # self.window.opts['center'] = view_location
        # self.window.setCameraPosition(distance=200., elevation=30, azimuth=98)
        # self.window.setCameraPosition(distance=400., elevation=20, azimuth=np.degrees(-sf*self.uc))
        # self.window.setCameraPosition(distance=1000., elevation=20, azimuth=np.degrees(sf*self.uc))
        # self.uc += 1

        self.app.processEvents()  # redraw

    def draw_agents(self, agent_states):
        """Updates all agents using the state message

        Parameters
        ----------
        agent_states : np.ndarray
            States of all agents where of size (#states x #agents).

        """
        for aa in range(self.num_agents):
            agent_xyz = np.array([[agent_states[0, aa]],
                                  [agent_states[1, aa]],
                                  [agent_states[2, aa]]])

            # attitude of agent as a rotation matrix R from body to inertial
            R = RotationBody2Vehicle(0., 0., agent_states[3, aa])

            # rotate and translate points defining agent
            if aa == P.main_rover_id:
                rotated_points = self.rotate_points(self.large_points, R)
                translated_points = self.translate_points(rotated_points, agent_xyz)
                face_names = ["large_rover_body_faces.csv",
                              "large_rover_wheels_faces.csv",
                              "large_rover_struts_faces.csv",
                              ]
                mesh = self.points_to_mesh(translated_points, face_names)
            else:
                rotated_points = self.rotate_points(self.agent_points, R)
                translated_points = self.translate_points(rotated_points, agent_xyz)
                face_names = ["small_rover_body_faces.csv",
                              "small_rover_wheels_faces.csv"]
                mesh = self.points_to_mesh(translated_points, face_names)

            # # convert North-East Down to East-North-Up for rendering
            # R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            # translated_points = np.matmul(R, translated_points)

            # convert points to triangular mesh defined as array of three 3D points (Nx3x3)

            if not self.agents_initialized:
                # initialize drawing of triangular mesh.
                self.agents.append(gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                                 vertexColors=self.agent_mesh_colors[aa],  # defines mesh colors (Nx1)
                                                 drawEdges=False,  # draw edges between mesh elements
                                                 smooth=False,  # speeds up rendering
                                                 computeNormals=False))  # speeds up rendering
                self.window.addItem(self.agents[aa])  # add body to plot
            else:
                # draw MAV by resetting mesh using rotated and translated points
                self.agents[aa].setMeshData(vertexes=mesh, vertexColors=self.agent_mesh_colors[aa])
                self.agents[aa].setDepthValue(0)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = np.matmul(R, points)
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_agent_points(self, vertex_names):
        # points are in NED coordinates
        points = np.zeros((0, 3))
        for vertex_name in vertex_names:
            new_vertices = np.genfromtxt('./meshes/' + vertex_name, delimiter=",")
            points = np.vstack((points, new_vertices))
        points = points.T

        # swap x and y axes
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        points = np.matmul(R, points)

        if len(vertex_names) == 2:
            points *= 5.

        return points

    def get_agent_colors(self, face_counts):
        """Get agent colors

        Parameters
        ----------
        face_counts : list of ints
            [number of body faces, number of wheel faces,
            number of strut faces]

        """
        #   define the colors for each face of triangular mesh
        white_gray = np.array([0.9, 0.9, 0.9, 1])
        dark_gray = np.array([0.3, 0.3, 0.3, 1])

        agent_mesh_colors = []

        for aa in range(self.num_agents):
            meshColors = np.empty((sum(face_counts), 3, 4), dtype=np.float32)
            meshColors[:face_counts[0]] = self.agent_colors[aa]  # body
            meshColors[face_counts[0]:face_counts[0] + face_counts[1]] = white_gray  # wheels
            meshColors[face_counts[0] + face_counts[1]:] = dark_gray  # wheels
            agent_mesh_colors.append(meshColors)

        return agent_mesh_colors

    def points_to_mesh(self, points, face_names):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)

        Parameters
        ----------
        points : np.ndarray
            shape of 3 x n
        face_names : list
            list of face names to load

        """
        points = points.T
        total_faces = 0
        face_counts = []
        full_mesh = np.zeros((0, 3, 3))
        for face_name in face_names:
            face_ids = np.genfromtxt("./meshes/" + face_name, delimiter=",")
            face_ids += total_faces
            face_array = np.array(list(map(lambda x: list(map(lambda y: points[int(y)], x)), face_ids)))
            total_faces = np.max(face_ids) + 1
            face_counts.append(face_array.shape[0])
            full_mesh = np.vstack((full_mesh, face_array))

        self.agent_mesh_colors = self.get_agent_colors(face_counts)
        # full_mesh = np.vstack((body_array,wheel_array))

        return full_mesh

    def draw_plan(self, plans):
        """ Draws dashed lines for planned routes

        Parameters
        plans : list of np.ndarrays
            List of [#pts x 3] np.ndarrays. The list should contain a
            np.ndarray for each agent. The plan will contain #pts and
            have a value for x, y, and z.

        """
        if not self.plans_initialized:
            self.plans = []
            for aa in range(len(plans)):
                plan_color = tuple(self.agent_colors[aa])
                self.plans.append(gl.GLLinePlotItem(pos=plans[aa][:, :3],
                                                    color=plan_color,
                                                    width=3,
                                                    antialias=True,
                                                    mode='lines'))
                self.window.addItem(self.plans[aa])
            self.plans_initialized = True
        else:
            for aa in range(len(plans)):
                if len(plans[aa]) != 1:
                    self.plans[aa].setData(pos=plans[aa][:, :3])

    def draw_paths(self):
        """ Draws solid lines for traveled paths

        """
        if self.state_history.shape[2] < 1:
            return
        if not self.paths_initialized:
            self.paths = []
            self.estimates = []
            for aa in range(self.num_agents):
                path_color = tuple(self.agent_colors[aa])
                self.paths.append(gl.GLLinePlotItem(pos=self.state_history[:3, aa].T,
                                                    color=path_color,
                                                    width=5,
                                                    antialias=True,
                                                    glOptions="opaque",
                                                    mode='line_strip'))
                self.window.addItem(self.paths[aa])
                self.estimates.append(gl.GLLinePlotItem(pos=self.estimate_history[:3, aa].T,
                                                        color=path_color,
                                                        width=3,
                                                        antialias=True,
                                                        glOptions="opaque",
                                                        mode='lines'))
                self.window.addItem(self.estimates[aa])
            self.paths_initialized = True
        else:
            for aa in range(self.num_agents):
                self.paths[aa].setData(pos=self.state_history[:3, aa].T)
                self.estimates[aa].setData(pos=self.estimate_history[:3, aa].T)

    def draw_ellipses(self, covs):
        """Draw error ellipses

        Parameters
        covs : tuple
            tuple of covariances for each agent

        """
        for aa in range(self.num_agents):
            # ellipse color
            ellipse_color = tuple(self.agent_colors[aa])

            P = 0.95  # percentile
            npoints = 100  # number of ellipse points
            theta = np.linspace(0, 2. * np.pi, npoints)
            ellipse_points = np.zeros((npoints, 3))
            mean = self.estimate_history[:2, aa, -1]
            cov = covs[aa][:2, :2]
            R = np.sqrt(-2. * np.log(1 - P))
            pts = np.array([R * np.cos(theta), R * np.sin(theta)])
            ellipse_points[:, :2] = (la.sqrtm(cov).dot(pts) + mean.reshape(-1, 1)).T
            # points should end as n x 3 array
            ellipse = gl.GLLinePlotItem(pos=ellipse_points,
                                        color=ellipse_color,
                                        width=2,
                                        antialias=True,
                                        glOptions="opaque",
                                        mode='lines')
            ellipse.setDepthValue(10)
            self.window.addItem(ellipse)

    def draw_grid(self, xlim, ylim):
        x_start, x_end, dx = xlim
        y_start, y_end, dy = ylim
        x_spaces = np.linspace(x_start, x_end, int((x_end - x_start) / dx) + 1)
        y_spaces = np.linspace(y_start, y_end, int((y_end - y_start) / dy) + 1)

        el = len(y_spaces[::2])  # even length
        ol = len(y_spaces[1::2])  # odd length
        R = np.hstack((x_end * np.ones((el, 1)),
                       y_spaces[::2].reshape(-1, 1),
                       np.zeros((el, 1))))
        RU = np.hstack((x_end * np.ones((ol, 1)),
                        y_spaces[1::2].reshape(-1, 1),
                        np.zeros((ol, 1))))
        L = np.hstack((x_start * np.ones((ol, 1)),
                       y_spaces[1::2].reshape(-1, 1),
                       np.zeros((ol, 1))))
        LU = np.hstack((x_start * np.ones((ol, 1)),
                        y_spaces[2::2].reshape(-1, 1),
                        np.zeros((el - 1, 1))))
        x_pts_total = R.shape[0] + RU.shape[0] + L.shape[0] + LU.shape[0] + 1
        x_lines = np.zeros((x_pts_total, 3))
        x_lines[0, :] = [x_start, y_start, 0.]
        x_lines[1::4, :] = R
        x_lines[2::4, :] = RU
        x_lines[3::4, :] = L
        x_lines[4::4, :] = LU
        x_grid = gl.GLLinePlotItem(pos=x_lines, color=pg.glColor('w'), width=1.0)
        self.window.addItem(x_grid)

        el = len(x_spaces[::2])  # even length
        ol = len(x_spaces[1::2])  # odd length
        U = np.hstack((x_spaces[::2].reshape(-1, 1),
                       y_end * np.ones((el, 1)),
                       np.zeros((el, 1))))
        UR = np.hstack((x_spaces[1::2].reshape(-1, 1),
                        y_end * np.ones((ol, 1)),
                        np.zeros((ol, 1))))
        D = np.hstack((x_spaces[1::2].reshape(-1, 1),
                       y_start * np.ones((ol, 1)),
                       np.zeros((ol, 1))))
        LU = np.hstack((x_spaces[2::2].reshape(-1, 1),
                        y_start * np.ones((ol, 1)),
                        np.zeros((el - 1, 1))))
        y_pts_total = U.shape[0] + UR.shape[0] + D.shape[0] + LU.shape[0] + 1
        y_lines = np.zeros((y_pts_total, 3))
        y_lines[0, :] = [x_start, y_start, 0.]
        y_lines[1::4, :] = U
        y_lines[2::4, :] = UR
        y_lines[3::4, :] = D
        y_lines[4::4, :] = LU
        y_grid = gl.GLLinePlotItem(pos=y_lines, color=pg.glColor('w'), width=1.0)
        self.window.addItem(y_grid)

    def draw_map(self):
        # set background color
        # self.window.setBackgroundColor([10.,10.,125.])  # set background color to blue
        self.window.setBackgroundColor('k')  # set background color to blue

        # draw grid
        self.draw_grid((P.grid_x[0], P.grid_x[1], P.grid_x[2]),
                       (P.grid_y[0], P.grid_y[1], P.grid_y[2]))

        # draw coordinate system
        axis_length = P.ref_frame_length

        x_axis_pts = np.array([[0.0, 0.0, 0.0],
                               [axis_length, 0.0, 0.0]])
        x_axis = gl.GLLinePlotItem(pos=x_axis_pts,
                                   color=pg.glColor('r'), width=3.0)
        self.window.addItem(x_axis)
        y_axis_pts = np.array([[0.0, 0.0, 0.0],
                               [0.0, axis_length, 0.0]])
        y_axis = gl.GLLinePlotItem(pos=y_axis_pts,
                                   color=pg.glColor('g'), width=3.0)
        self.window.addItem(y_axis)
        # z_axis_pts = np.array([[0.0,0.0,0.0],
        #                        [0.0,0.0,axis_length]])
        # z_axis = gl.GLLinePlotItem(pos=z_axis_pts,
        #                            color=pg.glColor('b'), width=3.0)
        # self.window.addItem(z_axis)

        for aa in range(self.num_agents):
            sphere = gl.MeshData.sphere(rows=10, cols=10, radius=2.0)
            goal_aa = gl.GLMeshItem(
                meshdata=sphere,
                color=self.agent_colors[aa],
                # shader="balloon",
                glOptions="additive",
                drawEdges=False,  # draw edges between mesh elements
                smooth=False,  # speeds up rendering
                computeNormals=False)  # speeds up rendering
            goal_aa.translate(*P.goal_states[:3, aa])
            self.window.addItem(goal_aa)
