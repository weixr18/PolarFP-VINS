/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include "feature_tracker.h"

void GlobalFeaturePool::beginFrame()
{
    cur_globals.clear();
    grid.clear();
}

void GlobalFeaturePool::propagateTracked(const std::vector<ChannelState>& channels)
{
    for (const auto& [global_id, channel_map] : prev_bindings) {
        for (const auto& [ch_name, prev_local_id] : channel_map) {
            const ChannelState* ch_ptr = nullptr;
            for (const auto& ch : channels) {
                if (ch.name == ch_name) {
                    ch_ptr = &ch;
                    break;
                }
            }
            if (!ch_ptr) continue;

            const auto& ch = *ch_ptr;
            auto it = std::find(ch.local_ids.begin(), ch.local_ids.end(), prev_local_id);
            if (it != ch.local_ids.end()) {
                size_t idx = std::distance(ch.local_ids.begin(), it);
                cur_globals[global_id].global_id = global_id;
                cur_globals[global_id].local_ids[ch_name] = prev_local_id;
                cur_globals[global_id].pixel_pts[ch_name] = ch.cur_pts[idx];
                insertToGrid(global_id, ch.cur_pts[idx]);
            }
        }
    }
}

void GlobalFeaturePool::registerUnboundFeatures(std::vector<ChannelState>& channels, int min_dist)
{
    for (auto& ch : channels) {
        for (size_t i = 0; i < ch.local_ids.size(); ++i) {
            int local_id = ch.local_ids[i];
            bool already_bound = false;
            for (const auto& [gid, gf] : cur_globals) {
                auto it = gf.local_ids.find(ch.name);
                if (it != gf.local_ids.end() && it->second == local_id) {
                    already_bound = true;
                    break;
                }
            }
            if (already_bound) continue;

            cv::Point2f pt = ch.cur_pts[i];
            auto candidates = queryNearby(pt);

            bool attached = false;
            for (int gid : candidates) {
                auto& gf = cur_globals[gid];
                if (gf.local_ids.count(ch.name)) continue;
                if (gf.pixel_pts.empty()) continue;

                double min_d = std::numeric_limits<double>::max();
                for (const auto& [_, other_pt] : gf.pixel_pts) {
                    double dx = pt.x - other_pt.x;
                    double dy = pt.y - other_pt.y;
                    min_d = std::min(min_d, dx*dx + dy*dy);
                }
                if (min_d <= (min_dist / 2.0) * (min_dist / 2.0)) {
                    gf.local_ids[ch.name] = local_id;
                    gf.pixel_pts[ch.name] = pt;
                    insertToGrid(gid, pt);
                    attached = true;
                    break;
                }
            }

            if (!attached) {
                int new_gid = next_global_id++;
                GlobalFeature gf;
                gf.global_id = new_gid;
                gf.local_ids[ch.name] = local_id;
                gf.pixel_pts[ch.name] = pt;
                cur_globals[new_gid] = std::move(gf);
                insertToGrid(new_gid, pt);
            }
        }
    }
}

std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
GlobalFeaturePool::buildFeatureFrame(const std::vector<ChannelState>& channels) const
{
    std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;

    std::map<std::string, const ChannelState*> ch_map;
    for (const auto& ch : channels) ch_map[ch.name] = &ch;

    for (const auto& [gid, gf] : cur_globals) {
        const ChannelState* selected_ch = nullptr;
        int selected_local_id = -1;
        size_t selected_idx = 0;

        for (const auto& ch : channels) {
            auto it = gf.local_ids.find(ch.name);
            if (it != gf.local_ids.end()) {
                selected_local_id = it->second;
                auto ch_it = ch_map.find(ch.name);
                if (ch_it == ch_map.end()) continue;
                selected_ch = ch_it->second;
                for (size_t i = 0; i < selected_ch->local_ids.size(); ++i) {
                    if (selected_ch->local_ids[i] == selected_local_id) {
                        selected_idx = i;
                        break;
                    }
                }
                break;
            }
        }

        if (!selected_ch) continue;

        const auto& ch = *selected_ch;
        double x = ch.cur_un_pts[selected_idx].x;
        double y = ch.cur_un_pts[selected_idx].y;
        double z = 1;
        double p_u = ch.cur_pts[selected_idx].x;
        double p_v = ch.cur_pts[selected_idx].y;
        double velocity_x = ch.pts_velocity[selected_idx].x;
        double velocity_y = ch.pts_velocity[selected_idx].y;

        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[gid].emplace_back(0, xyz_uv_velocity);
    }
    return featureFrame;
}

std::map<int, std::vector<std::pair<std::string, int>>>
GlobalFeaturePool::getGlobalToLocalMap() const
{
    std::map<int, std::vector<std::pair<std::string, int>>> result;
    for (const auto& [gid, gf] : cur_globals) {
        for (const auto& [ch_name, local_id] : gf.local_ids) {
            result[gid].emplace_back(ch_name, local_id);
        }
    }
    return result;
}

void GlobalFeaturePool::endFrame()
{
    prev_bindings.clear();
    prev_pts.clear();
    for (const auto& [gid, gf] : cur_globals) {
        if (!gf.local_ids.empty()) {
            prev_bindings[gid] = gf.local_ids;
            prev_pts[gid] = gf.pixel_pts;
        }
    }
}

std::pair<int, int> GlobalFeaturePool::getGridCell(const cv::Point2f& pt) const
{
    return {static_cast<int>(pt.x) / grid_size_, static_cast<int>(pt.y) / grid_size_};
}

void GlobalFeaturePool::insertToGrid(int global_id, const cv::Point2f& pt)
{
    auto cell = getGridCell(pt);
    grid[cell].push_back(global_id);
}

std::vector<int> GlobalFeaturePool::queryNearby(const cv::Point2f& pt) const
{
    std::vector<int> result;
    auto [cx, cy] = getGridCell(pt);
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            auto it = grid.find({cx + dx, cy + dy});
            if (it != grid.end()) {
                result.insert(result.end(), it->second.begin(), it->second.end());
            }
        }
    }
    return result;
}
