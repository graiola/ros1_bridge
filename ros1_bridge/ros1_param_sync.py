#!/usr/bin/env python3

import argparse
import fnmatch
import json
import os
import sys
import time
import xmlrpc.client
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node

from rcl_interfaces.msg import Parameter as RosParameterMsg
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters


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


def ros1_get_param_names(caller_id: str = "/ros1_param_sync") -> List[str]:
    master = get_ros1_master()
    names = xmlrpc_call(master, "getParamNames", caller_id)
    return sorted(names)


def ros1_get_param(name: str, caller_id: str = "/ros1_param_sync") -> Any:
    master = get_ros1_master()
    return xmlrpc_call(master, "getParam", caller_id, name)


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


def flatten_dict(prefix: str, value: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            new_key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(new_key, v))
    else:
        out[prefix] = value
    return out


def normalize_ros2_param_name(
    ros1_param_name: str,
    target_node: str,
    keep_slashes: bool = False,
    keep_leading_slash: bool = False,
    mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    Convert a ROS1 parameter path into a ROS2 parameter name.

    The expected symmetric case is:
      target_node = /ras_1/wolf_controller
      ros1_param_name = /ras_1/wolf_controller/robot_foot_names
    -> robot_foot_names

    Another example:
      /ras_1/wolf_controller/planner/max_vel
    -> planner.max_vel   (unless keep_slashes=True)

    If the ROS1 param is outside the target node namespace, we still normalize it.
    """
    mapping = mapping or {}

    # Prefer mapping on relative suffix when possible
    relative = ros1_param_name
    target_prefix = target_node.rstrip("/")
    if target_prefix and ros1_param_name == target_prefix:
        relative = ""
    elif target_prefix and ros1_param_name.startswith(target_prefix + "/"):
        relative = ros1_param_name[len(target_prefix) + 1 :]

    relative = relative.strip("/")

    if relative in mapping:
        return mapping[relative]

    if ros1_param_name in mapping:
        return mapping[ros1_param_name]

    name = relative if relative else ros1_param_name.strip("/")

    if not keep_leading_slash:
        name = name.lstrip("/")

    if not keep_slashes:
        name = name.replace("/", ".")

    while ".." in name:
        name = name.replace("..", ".")

    return name.strip(".")


def homogeneous_list_type(values: List[Any]):
    if not values:
        return None
    first_t = type(values[0])
    if all(isinstance(v, first_t) for v in values):
        return first_t
    return None


def python_value_to_parameter_value(value: Any) -> ParameterValue:
    pv = ParameterValue()

    if isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = bool(value)
        return pv

    if isinstance(value, int) and not isinstance(value, bool):
        pv.type = ParameterType.PARAMETER_INTEGER
        pv.integer_value = int(value)
        return pv

    if isinstance(value, float):
        pv.type = ParameterType.PARAMETER_DOUBLE
        pv.double_value = float(value)
        return pv

    if isinstance(value, str):
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = str(value)
        return pv

    if isinstance(value, list):
        if len(value) == 0:
            pv.type = ParameterType.PARAMETER_STRING_ARRAY
            pv.string_array_value = []
            return pv

        t = homogeneous_list_type(value)
        if t is bool:
            pv.type = ParameterType.PARAMETER_BOOL_ARRAY
            pv.bool_array_value = [bool(v) for v in value]
            return pv
        if t is int:
            pv.type = ParameterType.PARAMETER_INTEGER_ARRAY
            pv.integer_array_value = [int(v) for v in value]
            return pv
        if t is float:
            pv.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            pv.double_array_value = [float(v) for v in value]
            return pv
        if t is str:
            pv.type = ParameterType.PARAMETER_STRING_ARRAY
            pv.string_array_value = [str(v) for v in value]
            return pv

        # Mixed arrays are not valid ROS2 parameter arrays
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = json.dumps(value)
        return pv

    pv.type = ParameterType.PARAMETER_STRING
    pv.string_value = json.dumps(value)
    return pv


def to_parameter_msgs(name: str, value: Any) -> List[RosParameterMsg]:
    if isinstance(value, dict):
        flat = flatten_dict(name, value)
        out: List[RosParameterMsg] = []
        for k, v in flat.items():
            out.extend(to_parameter_msgs(k, v))
        return out

    msg = RosParameterMsg()
    msg.name = name
    msg.value = python_value_to_parameter_value(value)
    return [msg]


class Ros1ParamSyncNode(Node):
    def __init__(self):
        super().__init__("ros1_param_sync")

    def wait_for_set_parameters_service(self, target_node: str, timeout_sec: float):
        service_name = f"{target_node.rstrip('/')}/set_parameters"
        client = self.create_client(SetParameters, service_name)
        available = client.wait_for_service(timeout_sec=timeout_sec)
        return client if available else None


def parse_mapping(entries: List[str]) -> Dict[str, str]:
    """
    Mapping applies primarily to the ROS1 suffix under the target node.

    Examples:
      robot_description:=robot_description
      planner/max_vel:=planner.max_vel

    Absolute ROS1 parameter names are also accepted as keys.
    """
    result: Dict[str, str] = {}
    for item in entries:
        if ":=" not in item:
            raise ValueError(f"Invalid mapping '{item}', expected ROS1_SUFFIX:=ROS2_PARAM")
        src, dst = item.split(":=", 1)
        result[src.strip()] = dst.strip()
    return result


def get_full_node_names(node: Ros1ParamSyncNode) -> List[str]:
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


def list_matching_target_nodes(node: Ros1ParamSyncNode, pattern: str) -> List[str]:
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


def wait_for_matching_target_nodes(
    node: Ros1ParamSyncNode,
    pattern: str,
    timeout_sec: float,
    poll_period_sec: float = 0.5,
) -> List[str]:
    deadline = time.monotonic() + timeout_sec
    last_seen: List[str] = []

    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

        matches = list_matching_target_nodes(node, pattern)
        if matches:
            return matches

        last_seen = get_full_node_names(node)
        time.sleep(poll_period_sec)

    raise RuntimeError(
        f"No ROS2 target nodes matched pattern '{pattern}' within {timeout_sec:.1f}s. "
        f"Visible nodes at timeout: {last_seen}"
    )


def resolve_ros1_param_names_for_target(
    target_node: str,
    use_all: bool,
    explicit_names: Optional[List[str]],
    prefix: Optional[str],
) -> List[str]:
    """
    Build the ROS1 parameter names to read for a given target node.

    Symmetric behavior:
      target_node = /ras_1/wolf_controller
      --params robot_foot_names planner/max_vel
    =>
      /ras_1/wolf_controller/robot_foot_names
      /ras_1/wolf_controller/planner/max_vel

    Absolute names are preserved as-is.
    """
    if use_all:
        all_names = ros1_get_param_names()

        base_prefix = target_node.rstrip("/")
        if prefix:
            base_prefix = join_ros_names(base_prefix, prefix)

        return [n for n in all_names if n == base_prefix or n.startswith(base_prefix + "/")]

    names: List[str] = []
    for item in explicit_names or []:
        if item.startswith("/"):
            names.append(item)
        else:
            names.append(join_ros_names(target_node, item))
    return names


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sync ROS1 parameters into ROS2 node parameters"
    )
    parser.add_argument(
        "--target-node",
        required=True,
        help=(
            "ROS2 target node name or wildcard pattern, e.g. /wolf_controller "
            "or '/*/wolf_controller'"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Sync all ROS1 parameters under each target node namespace",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        help=(
            "Explicit ROS1 parameter suffixes to sync relative to the target node. "
            "Absolute ROS1 parameter names are also accepted."
        ),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional ROS1 parameter prefix under the target node when using --all",
    )
    parser.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=[],
        help="Name remap ROS1_SUFFIX:=ROS2_PARAM ; can be repeated",
    )
    parser.add_argument(
        "--keep-slashes",
        action="store_true",
        help="Do not convert '/' to '.' in ROS2 parameter names",
    )
    parser.add_argument(
        "--keep-leading-slash",
        action="store_true",
        help="Do not strip leading '/' from fallback normalized names",
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
        help="Print parameters without setting them",
    )

    args = parser.parse_args(argv)

    if args.all and args.params:
        parser.error("Use either --all or --params, not both")
    if not args.all and not args.params:
        parser.error("Use either --all or --params")

    mapping = parse_mapping(args.maps)

    rclpy.init()
    node = Ros1ParamSyncNode()

    try:
        target_nodes = wait_for_matching_target_nodes(
            node=node,
            pattern=args.target_node,
            timeout_sec=args.timeout,
        )

        node.get_logger().info(
            "Matched target node(s): " + ", ".join(target_nodes)
        )

        total_ok = 0
        total_fail = 0

        for target_node in target_nodes:
            node.get_logger().info(f"Processing target node {target_node}")

            try:
                ros1_param_names = resolve_ros1_param_names_for_target(
                    target_node=target_node,
                    use_all=args.all,
                    explicit_names=args.params,
                    prefix=args.prefix,
                )

                if not ros1_param_names:
                    node.get_logger().warning(
                        f"No ROS1 parameters matched the request for {target_node}"
                    )
                    continue

                expanded: List[RosParameterMsg] = []

                for ros1_name in ros1_param_names:
                    try:
                        value = ros1_get_param(ros1_name)
                        ros2_name = normalize_ros2_param_name(
                            ros1_param_name=ros1_name,
                            target_node=target_node,
                            keep_slashes=args.keep_slashes,
                            keep_leading_slash=args.keep_leading_slash,
                            mapping=mapping,
                        )
                        expanded.extend(to_parameter_msgs(ros2_name, value))
                    except Exception as exc:
                        node.get_logger().warning(
                            f"Failed converting ROS1 param for {target_node}: {ros1_name}: {exc}"
                        )

                print(f"Parameters to sync into {target_node}:")
                for p in expanded:
                    print(f"  {p.name} = {p.value}")

                if not expanded:
                    node.get_logger().warning(
                        f"No converted parameters available for {target_node}"
                    )
                    continue

                if args.dry_run:
                    continue

                client = node.wait_for_set_parameters_service(target_node, args.timeout)
                if client is None:
                    raise RuntimeError(
                        f"ROS2 set_parameters service for target node '{target_node}' is not available"
                    )

                request = SetParameters.Request()
                request.parameters = expanded

                future = client.call_async(request)
                rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)

                if not future.done():
                    raise TimeoutError(
                        f"Timed out waiting for set_parameters response from {target_node}"
                    )

                response = future.result()
                if response is None:
                    raise RuntimeError(
                        f"{target_node}/set_parameters returned no response"
                    )

                ok = 0
                fail = 0
                for param, result in zip(expanded, response.results):
                    if result.successful:
                        ok += 1
                        node.get_logger().info(f"Set {target_node}:{param.name}")
                    else:
                        fail += 1
                        node.get_logger().error(
                            f"Failed {target_node}:{param.name}: {result.reason}"
                        )

                total_ok += ok
                total_fail += fail

            except Exception as exc:
                total_fail += 1
                node.get_logger().error(f"Failed processing {target_node}: {exc}")

        node.get_logger().info(
            f"Finished: successful={total_ok} failed={total_fail}"
        )
        return 0 if total_fail == 0 else 1

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
