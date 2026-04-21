#!/usr/bin/env python3

import argparse
import fnmatch
import os
import time
import xmlrpc.client
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, ListParameters


def xmlrpc_call(proxy, method: str, *args):
    fn = getattr(proxy, method)
    code, msg, value = fn(*args)
    if code != 1:
        raise RuntimeError(f"ROS1 XML-RPC call failed: {method}: {msg}")
    return value


def get_ros1_master() -> xmlrpc.client.ServerProxy:
    uri = os.environ.get("ROS_MASTER_URI")
    if not uri:
        raise RuntimeError("ROS_MASTER_URI is not set")
    return xmlrpc.client.ServerProxy(uri)


def ros1_set_param(name: str, value, caller_id: str = "/ros2_param_sync"):
    master = get_ros1_master()
    return xmlrpc_call(master, "setParam", caller_id, name, value)


def normalize_ros1_param_leaf_name(
    ros2_param_name: str,
    keep_dots: bool = False,
    mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convert a ROS 2 parameter name into the leaf/path part appended under the source node namespace.

    Examples:
      robot_foot_names       -> robot_foot_names
      planner.max_vel        -> planner/max_vel   (unless keep_dots=True)
    """
    mapping = mapping or {}
    if ros2_param_name in mapping:
        mapped = mapping[ros2_param_name].strip()
        return mapped.lstrip("/")

    name = ros2_param_name.strip()
    if not keep_dots:
        name = name.replace(".", "/")

    while "//" in name:
        name = name.replace("//", "/")

    return name.strip("/")


def join_ros_names(prefix: str, leaf: str) -> str:
    prefix = prefix.strip()
    leaf = leaf.strip()

    if not prefix.startswith("/"):
        prefix = "/" + prefix if prefix else ""

    prefix = prefix.rstrip("/")
    leaf = leaf.lstrip("/")

    if not prefix and not leaf:
        return "/"
    if not prefix:
        return "/" + leaf
    if not leaf:
        return prefix
    return f"{prefix}/{leaf}"


def parameter_value_to_python(pv):
    t = pv.type

    if t == ParameterType.PARAMETER_BOOL:
        return pv.bool_value
    if t == ParameterType.PARAMETER_INTEGER:
        return pv.integer_value
    if t == ParameterType.PARAMETER_DOUBLE:
        return pv.double_value
    if t == ParameterType.PARAMETER_STRING:
        return pv.string_value
    if t == ParameterType.PARAMETER_BOOL_ARRAY:
        return list(pv.bool_array_value)
    if t == ParameterType.PARAMETER_INTEGER_ARRAY:
        return list(pv.integer_array_value)
    if t == ParameterType.PARAMETER_DOUBLE_ARRAY:
        return list(pv.double_array_value)
    if t == ParameterType.PARAMETER_STRING_ARRAY:
        return list(pv.string_array_value)

    # Skip PARAMETER_NOT_SET and unsupported values
    return None


class Ros2ParamSyncNode(Node):
    def __init__(self):
        super().__init__("ros2_param_sync")

    def wait_for_list_client(self, source_node: str, timeout_sec: float):
        service_name = f"{source_node.rstrip('/')}/list_parameters"
        client = self.create_client(ListParameters, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return None
        return client

    def wait_for_get_client(self, source_node: str, timeout_sec: float):
        service_name = f"{source_node.rstrip('/')}/get_parameters"
        client = self.create_client(GetParameters, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return None
        return client


def parse_mapping(entries: List[str]) -> Dict[str, str]:
    """
    Mapping applies to the parameter leaf name, not to the source node namespace.

    Example:
      planner.max_vel:=planner/max_vel
      robot_description:=description
    """
    result: Dict[str, str] = {}
    for item in entries:
        if ":=" not in item:
            raise ValueError(f"Invalid mapping '{item}', expected ROS2_PARAM:=ROS1_PARAM_SUFFIX")
        src, dst = item.split(":=", 1)
        result[src.strip()] = dst.strip()
    return result


def get_full_node_names(node: Ros2ParamSyncNode) -> List[str]:
    names_and_namespaces = node.get_node_names_and_namespaces()
    full_names: List[str] = []

    for name, namespace in names_and_namespaces:
        ns = namespace or "/"
        if not ns.startswith("/"):
            ns = "/" + ns
        ns = ns.rstrip("/")
        if ns == "":
            ns = "/"

        if ns == "/":
            full_name = f"/{name}"
        else:
            full_name = f"{ns}/{name}"

        full_names.append(full_name)

    return sorted(set(full_names))


def expand_optional_namespace_patterns(pattern: str) -> List[str]:
    """
    Allow each '/*/' segment to match either:
      - one namespace level
      - zero namespace levels
    """
    candidate_patterns = {pattern}
    queue = [pattern]

    while queue:
        current = queue.pop()
        idx = current.find("/*/")
        while idx != -1:
            collapsed = current[:idx] + "/" + current[idx + len("/*/"):]
            while "//" in collapsed:
                collapsed = collapsed.replace("//", "/")
            if collapsed not in candidate_patterns:
                candidate_patterns.add(collapsed)
                queue.append(collapsed)
            idx = current.find("/*/", idx + 1)

    return sorted(candidate_patterns)


def list_matching_source_nodes(node: Ros2ParamSyncNode, pattern: str) -> List[str]:
    """
    Resolve a source node argument that may contain wildcards.

    Matching is performed on fully qualified ROS node names, e.g.:
      /wolf_controller
      /ras_1/wolf_controller
      /ras_2/wolf_controller

    Special behavior:
      - patterns like '/*/wolf_controller' also match '/wolf_controller'
      - each '/*/' can match either one namespace level or zero levels
    """
    full_names = get_full_node_names(node)

    if not any(ch in pattern for ch in ["*", "?", "["]):
        return [pattern] if pattern in full_names else []

    candidate_patterns = expand_optional_namespace_patterns(pattern)

    matches: List[str] = []
    for full_name in full_names:
        for candidate in candidate_patterns:
            if fnmatch.fnmatch(full_name, candidate):
                matches.append(full_name)
                break

    return sorted(set(matches))


def wait_for_matching_source_nodes(
    node: Ros2ParamSyncNode,
    pattern: str,
    timeout_sec: float,
    poll_period_sec: float = 0.5,
) -> List[str]:
    """
    Wait until at least one ROS2 node matches the source pattern.
    Return an empty list on timeout instead of raising, so callers can warn and continue.
    """
    deadline = time.monotonic() + timeout_sec
    last_seen: List[str] = []

    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

        matches = list_matching_source_nodes(node, pattern)
        if matches:
            return matches

        last_seen = get_full_node_names(node)
        time.sleep(poll_period_sec)

    node.get_logger().warning(
        f"No ROS2 nodes matched source pattern '{pattern}' within {timeout_sec:.1f}s. "
        f"Visible nodes at timeout: {last_seen}"
    )
    return []


def list_ros2_parameters(
    node: Ros2ParamSyncNode,
    source_node: str,
    prefix: Optional[str],
    timeout: float
) -> List[str]:
    client = node.wait_for_list_client(source_node, timeout)
    if client is None:
        raise RuntimeError(f"Could not reach {source_node}/list_parameters")

    req = ListParameters.Request()
    req.prefixes = [prefix] if prefix else []
    req.depth = ListParameters.Request.DEPTH_RECURSIVE

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    if not future.done():
        raise TimeoutError(f"Timed out waiting for {source_node}/list_parameters")

    resp = future.result()
    if resp is None:
        raise RuntimeError(f"{source_node}/list_parameters returned no response")

    return list(resp.result.names)


def get_ros2_parameters(
    node: Ros2ParamSyncNode,
    source_node: str,
    names: List[str],
    timeout: float
):
    client = node.wait_for_get_client(source_node, timeout)
    if client is None:
        raise RuntimeError(f"Could not reach {source_node}/get_parameters")

    req = GetParameters.Request()
    req.names = names

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    if not future.done():
        raise TimeoutError(f"Timed out waiting for {source_node}/get_parameters")

    resp = future.result()
    if resp is None:
        raise RuntimeError(f"{source_node}/get_parameters returned no response")

    return resp.values


def select_parameter_names(
    all_names: List[str],
    explicit_names: Optional[List[str]],
    use_all: bool
) -> List[str]:
    if use_all:
        return all_names
    return explicit_names or []


def build_ros1_param_name(
    source_node: str,
    ros2_param_name: str,
    keep_dots: bool,
    mapping: Dict[str, str],
) -> str:
    suffix = normalize_ros1_param_leaf_name(
        ros2_param_name,
        keep_dots=keep_dots,
        mapping=mapping,
    )
    return join_ros_names(source_node, suffix)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sync ROS2 node parameters into the ROS1 parameter server"
    )
    parser.add_argument(
        "--source-node",
        required=True,
        help=(
            "ROS2 source node name or wildcard pattern, e.g. /wolf_controller "
            "or '/*/wolf_controller'"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all parameters from the ROS2 node(s)",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help="Explicit ROS2 parameter names to export",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional ROS2 parameter prefix filter when using --all",
    )
    parser.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=[],
        help="Name remap ROS2_PARAM:=ROS1_PARAM_SUFFIX ; can be repeated",
    )
    parser.add_argument(
        "--keep-dots",
        action="store_true",
        help="Keep dots in parameter suffixes instead of converting '.' to '/'",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout for ROS2 graph and service calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parameters without writing them to ROS1",
    )

    args = parser.parse_args(argv)

    if args.all and args.params:
        parser.error("Use either --all or --params, not both")
    if not args.all and not args.params:
        parser.error("Use either --all or --params")

    mapping = parse_mapping(args.maps)

    rclpy.init()
    node = Ros2ParamSyncNode()

    try:
        source_nodes = wait_for_matching_source_nodes(
            node=node,
            pattern=args.source_node,
            timeout_sec=args.timeout,
        )

        if not source_nodes:
            node.get_logger().warning(
                f"Skipping sync because no ROS2 source nodes matched '{args.source_node}'"
            )
            return 0

        node.get_logger().info(
            "Matched source node(s): " + ", ".join(source_nodes)
        )

        total_ok = 0
        total_fail = 0

        for source_node in source_nodes:
            node.get_logger().info(f"Processing source node {source_node}")

            try:
                all_names = list_ros2_parameters(node, source_node, args.prefix, args.timeout)
                selected_names = select_parameter_names(all_names, args.params, args.all)

                if not selected_names:
                    node.get_logger().warning(
                        f"No ROS2 parameters matched the request for {source_node}"
                    )
                    continue

                values = get_ros2_parameters(node, source_node, selected_names, args.timeout)

                export_pairs: List[Tuple[str, str, object]] = []
                for ros2_name, value_msg in zip(selected_names, values):
                    value = parameter_value_to_python(value_msg)
                    if value is None:
                        node.get_logger().warning(
                            f"Skipping unset/unsupported parameter on {source_node}: {ros2_name}"
                        )
                        continue

                    ros1_name = build_ros1_param_name(
                        source_node=source_node,
                        ros2_param_name=ros2_name,
                        keep_dots=args.keep_dots,
                        mapping=mapping,
                    )
                    export_pairs.append((ros2_name, ros1_name, value))

                print(f"Parameters to sync from {source_node}:")
                for ros2_name, ros1_name, value in export_pairs:
                    print(f"  {ros2_name} -> {ros1_name} = {value!r}")

                if args.dry_run:
                    continue

                ok = 0
                fail = 0
                for ros2_name, ros1_name, value in export_pairs:
                    try:
                        ros1_set_param(ros1_name, value)
                        ok += 1
                        node.get_logger().info(
                            f"Set ROS1 param {ros1_name} from {source_node}:{ros2_name}"
                        )
                    except Exception as exc:
                        fail += 1
                        node.get_logger().error(
                            f"Failed {source_node}:{ros2_name} -> {ros1_name}: {exc}"
                        )

                total_ok += ok
                total_fail += fail

            except Exception as exc:
                total_fail += 1
                node.get_logger().error(f"Failed processing {source_node}: {exc}")

        node.get_logger().info(
            f"Finished: successful={total_ok} failed={total_fail}"
        )
        return 0 if total_fail == 0 else 1

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
